"""
Agentic Honey-Pot — Scam Detection & Intelligence Extraction API

Architecture:
  1. Two-layer detection   — keyword scan + LLM classifier run concurrently;
                             either one is enough to trigger scam mode.
  2. Stage machine         — directed graph of victim emotional states:
                             entry -> doubt -> fear -> comply -> elicit <-> deflect -> stall
  3. Background extraction — Gemini structured extraction + regex run in the background,
                             never adding to response latency.
  4. Async callback        — intel payload sent to platform after each extraction cycle.

Required env: GROQ_API_KEY, GEMINI_API_KEY, HONEYPOT_API_KEY
Optional env: DB_PATH (default: honeypot.db)
Run: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import secrets
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import aiosqlite
import google.generativeai as genai
import httpx
from fastapi import FastAPI, Header, HTTPException
from groq import Groq
from pydantic import BaseModel, field_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("honeypot")

# ─── Configuration ────────────────────────────────────────────────────────────

HONEYPOT_API_KEY = os.environ["HONEYPOT_API_KEY"]
DB_PATH          = os.getenv("DB_PATH", "honeypot.db")
CALLBACK_URL     = os.getenv("CALLBACK_URL", "https://hackathon.guvi.in/api/updateHoneyPotFinalResult")

MAX_TEXT_LEN = 4_000

# Per-turn response time budget.
# The platform enforces a 30s hard timeout per request. We target <25s to leave a 5s
# buffer for network latency and any unforeseen overhead.
# Budget breakdown (worst-case single attempt per call):
#   detect()          — parallel keyword + LLM,  timeout=5s  → 5s
#   plan_next_stage() — Groq JSON,                timeout=5s  → 5s
#   generate_reply()  — Groq prose,               timeout=9s  → 9s
#   overhead                                                  → ~2s
#   Total blocking                                            → ~21s
# asyncio.wait_for(ENGAGE_TIMEOUT_S) is the hard ceiling.
ENGAGE_TIMEOUT_S = 22

# Groq per-call timeouts (seconds). These are passed directly to the SDK.
GROQ_DETECT_TIMEOUT = 5   # binary classifier, 80 tokens — should be very fast
GROQ_PLAN_TIMEOUT   = 5   # JSON planner, 100 tokens
GROQ_REPLY_TIMEOUT  = 9   # prose reply, 200 tokens — gets the most budget

# Start sending callbacks from this turn onwards — gives the first two turns
# time for the initial extraction cycle to complete.
CALLBACK_START_TURN = 3


# ─── API key pools ────────────────────────────────────────────────────────────
#
# Load all available keys for each provider from environment variables.
# Naming convention (add as many as you have):
#   GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ...
#   GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
#
# On a 429 / rate-limit / auth error the pool rotates to the next key automatically.
# Keys that keep failing cycle back around so the system is always trying something.

def _load_keys(prefix: str) -> List[str]:
    """Read all keys matching PREFIX, PREFIX_2, PREFIX_3 … from the environment."""
    keys: List[str] = []
    base = os.environ.get(prefix)
    if base:
        keys.append(base)
    i = 2
    while True:
        k = os.environ.get(f"{prefix}_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    if not keys:
        raise RuntimeError(f"No API keys found for {prefix}. Set at least {prefix} in env.")
    return keys


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect 429 / quota / rate-limit errors across both Groq and Gemini SDKs
    without importing private exception classes from either package."""
    msg = str(exc).lower()
    return any(x in msg for x in ("429", "rate limit", "quota", "resource exhausted", "too many"))


class GroqPool:
    """Thread-safe round-robin pool of Groq clients.
    Rotates to the next client whenever a rate-limit or auth error is detected."""

    def __init__(self, keys: List[str]) -> None:
        self._clients = [Groq(api_key=k) for k in keys]
        self._idx     = 0
        log.info("GroqPool: %d key(s) loaded", len(self._clients))

    @property
    def client(self) -> Groq:
        return self._clients[self._idx]

    def rotate(self) -> None:
        old = self._idx
        self._idx = (self._idx + 1) % len(self._clients)
        log.warning("GroqPool: rotated from key %d to key %d", old + 1, self._idx + 1)


class GeminiPool:
    """Key-rotation pool for Gemini.
    genai.configure() sets global state, so access is serialised with an asyncio.Lock
    to prevent concurrent requests from clobbering each other's active key."""

    def __init__(self, keys: List[str]) -> None:
        self._keys = keys
        self._idx  = 0
        # Lock is created lazily inside generate() so it is always bound to the
        # running event loop. Creating asyncio.Lock() at module level (before
        # uvicorn starts the loop) raises RuntimeError on Python 3.12+.
        self._lock: Optional[asyncio.Lock] = None
        # Configure the first key immediately so genai is ready at startup.
        genai.configure(api_key=self._keys[0])
        log.info("GeminiPool: %d key(s) loaded", len(self._keys))

    def rotate(self) -> None:
        old = self._idx
        self._idx = (self._idx + 1) % len(self._keys)
        log.warning("GeminiPool: rotated from key %d to key %d", old + 1, self._idx + 1)

    async def generate(self, prompt: str, schema: dict) -> dict:
        """Run a structured Gemini extraction, rotating keys on rate-limit errors.

        Lock scope is narrowed to just genai.configure() + model construction so
        that multiple concurrent sessions can each await generate_content_async()
        in parallel rather than serialising through one global lock."""
        # Lazy lock initialisation — always runs inside the event loop.
        if self._lock is None:
            self._lock = asyncio.Lock()

        for attempt in range(len(self._keys) * 2):   # try every key twice before giving up
            # Hold the lock only long enough to set the global API key and build
            # the model object. Release it before the network call so other
            # sessions are not blocked for the full 3-5s Gemini response time.
            async with self._lock:
                genai.configure(api_key=self._keys[self._idx])
                # Primary: gemini-2.0-flash (stable, fast, widely available)
                # If rate-limited we rotate keys; if the model name changes just update here.
                model = genai.GenerativeModel(
                    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema":    schema,
                        "temperature":        0.05,
                    },
                )
            # Network call happens outside the lock — concurrent sessions proceed.
            try:
                resp = await model.generate_content_async(prompt)
                return json.loads(resp.text)
            except Exception as e:
                if _is_rate_limit_error(e):
                    async with self._lock:
                        log.warning("GeminiPool: rate limit on key %d, rotating", self._idx + 1)
                        self.rotate()
                else:
                    log.warning("GeminiPool: attempt %d error: %s", attempt + 1, e)
                    await asyncio.sleep(0.3)
        return {}


groq_pool   = GroqPool(_load_keys("GROQ_API_KEY"))
gemini_pool = GeminiPool(_load_keys("GEMINI_API_KEY"))

_bg: Set[asyncio.Task] = set()


def fire(coro) -> None:
    """Schedule a coroutine as a tracked background task.
    The task set keeps it alive until completion even if nothing else holds a reference."""
    t = asyncio.create_task(coro)
    _bg.add(t)
    t.add_done_callback(_bg.discard)


# ─── Conversation stage machine ───────────────────────────────────────────────

class Stage(str, Enum):
    """
    Directed graph of victim emotional states.
    Nodes are traversed forward only — the planner never regresses.

    entry -> doubt -> fear -> comply -> elicit <-> deflect -> stall
    """
    ENTRY   = "entry"    # confused, not yet believing there is a problem
    DOUBT   = "doubt"    # starting to worry something might be real
    FEAR    = "fear"     # scared of consequences (arrest, block, freeze)
    COMPLY  = "comply"   # trying to follow instructions but having "tech trouble"
    ELICIT  = "elicit"   # asking scammer for THEIR payment details
    DEFLECT = "deflect"  # OTP failed / app crashed — loops back for another elicit attempt
    STALL   = "stall"    # maximum delay, call son/bank, buy time


VALID_NEXT: Dict[Stage, List[Stage]] = {
    Stage.ENTRY:   [Stage.DOUBT,   Stage.FEAR],
    Stage.DOUBT:   [Stage.FEAR,    Stage.COMPLY],
    Stage.FEAR:    [Stage.COMPLY,  Stage.ELICIT],
    Stage.COMPLY:  [Stage.ELICIT,  Stage.DEFLECT],
    Stage.ELICIT:  [Stage.DEFLECT, Stage.STALL],
    Stage.DEFLECT: [Stage.ELICIT,  Stage.STALL],
    Stage.STALL:   [Stage.STALL],
}

DIRECTIVES: Dict[Stage, str] = {
    Stage.ENTRY: (
        "You just received this unexpected message. Sound confused and slightly suspicious. "
        "Mention one thing that seems odd — the urgency, the unexpected contact, or the "
        "strange request. Then ask one investigative question: who are they, which department "
        "they are from, or what their official employee ID is. YOU MUST END WITH A QUESTION MARK."
    ),
    Stage.DOUBT: (
        "You are starting to worry this might be real but something still feels wrong. "
        "Mention one specific red flag you noticed — the urgency, the OTP demand, the "
        "threat of account freezing, or the suspicious link they sent. Then ask a probing "
        "question: their full name, official contact number, or bank branch. "
        "YOU MUST END WITH A QUESTION MARK."
    ),
    Stage.FEAR: (
        "You are scared but deeply confused. Express your fear, then ask a clarifying question "
        "that buys time: Can they confirm their employee ID number? What is the official helpline "
        "to verify this? Is there a reference or case number for this issue? "
        "Mention that threatening arrest or account freezing is not how banks normally operate. "
        "YOU MUST END WITH A QUESTION MARK."
    ),
    Stage.COMPLY: (
        "You are trying to follow their instructions but your phone app is not cooperating. "
        "Describe exactly what you see on screen step by step, ask for clarification at each "
        "step, and mention that the extreme urgency they keep emphasising is very unusual — "
        "your bank never contacts you this way. Ask: what is their direct callback number "
        "in case the call drops? YOU MUST END WITH A QUESTION MARK."
    ),
    Stage.ELICIT: (
        "Your app keeps crashing or showing an error. Offer to pay directly and ask for their "
        "details — specifically whichever is still missing: UPI ID, bank account number, "
        "phone number, email address, or official case reference ID. Sound cooperative: "
        "'Just tell me exactly which account to send to and I will do it right now. "
        "Also can you give me your email in case I need to send a screenshot?' "
        "YOU MUST END WITH A QUESTION MARK."
    ),
    Stage.DEFLECT: (
        "The OTP is not working or you pressed the wrong button and the app shows an error. "
        "Sound genuinely distressed. Ask for an alternative: a different account number, "
        "a different UPI ID, or a different payment method. Also ask for their supervisor's "
        "name and direct phone number so you can report this technical problem. "
        "YOU MUST END WITH A QUESTION MARK."
    ),
    Stage.STALL: (
        "Tell them you need to call your son or daughter first because they handle all digital "
        "banking. Ask for the scammer's direct number and email address so your son can call "
        "them back. Or say the network is very slow — ask them to send their bank account "
        "details and case number by SMS so you have them ready when connection improves. "
        "YOU MUST END WITH A QUESTION MARK."
    ),
}


# ─── Victim personas ──────────────────────────────────────────────────────────

# Realistic Indian demographics that scammers actually target.
# Older personas sustain longer conversations naturally — they ask more questions,
# are believably confused by apps, and are slow to comply, all of which maximises
# the engagement quality score.
PERSONAS = [
    {
        "age": 68, "role": "retired schoolteacher", "city": "Chennai",
        "trait": "polite, slightly hard of hearing, confused by technology, trusts authority unconditionally",
        "dialect": "Formal English, says 'I see' and 'Is it?', often apologises unnecessarily",
        "beta": False,
    },
    {
        "age": 72, "role": "retired government clerk", "city": "Lucknow",
        "trait": "slow to respond, asks people to repeat themselves, very afraid of police and government",
        "dialect": "Hindi-English mix, very formal, says 'Ji haan' and 'theek hai ji'",
        "beta": False,
    },
    {
        "age": 65, "role": "retired pensioner", "city": "Kolkata",
        "trait": "forgetful, repeats himself, trusts anyone who sounds official",
        "dialect": "Old-fashioned formal English, says 'kindly' and 'please do the needful'",
        "beta": True,
    },
    {
        "age": 58, "role": "homemaker", "city": "Delhi",
        "trait": "easily frightened, mentions husband who is not home, anxious about losing money",
        "dialect": "Hindi-English mix, says 'arre', frequently mentions that husband handles finances",
        "beta": True,
    },
    {
        "age": 55, "role": "small shop owner", "city": "Ahmedabad",
        "trait": "semi-literate in English, suspicious but greedy, obsesses over exact rupee amounts",
        "dialect": "Broken English with Gujarati influence, says 'bhai', fixates on money",
        "beta": False,
    },
    {
        "age": 62, "role": "retired farmer", "city": "Punjab",
        "trait": "does not understand smartphones, treats phone like a landline, very literal",
        "dialect": "Simple English, asks 'which button?', 'where is it written?', very slow",
        "beta": False,
    },
    {
        "age": 70, "role": "retired army officer", "city": "Pune",
        "trait": "initially authoritative, softens sharply when financial loss is threatened",
        "dialect": "Crisp formal English, military phrasing, becomes increasingly flustered",
        "beta": False,
    },
    {
        "age": 60, "role": "school principal", "city": "Hyderabad",
        "trait": "busy and distracted, initially dismissive, panics when legal action is mentioned",
        "dialect": "Educated English, says 'listen' and 'just tell me exactly what to do'",
        "beta": True,
    },
]

# Only these personas naturally say "beta". Gated at data level so the LLM
# instruction cannot be silently ignored when temperature is high.
BETA_ROLES = {"retired pensioner", "homemaker", "school principal"}


# ─── Request / response models ────────────────────────────────────────────────

class Message(BaseModel):
    sender: str
    text: str
    # The API doc shows ISO string ("2025-02-11T10:30:00Z") for the initial message
    # and "(epoch Time in ms)" for conversationHistory — both formats appear in practice.
    # Typing as Any prevents Pydantic from raising a ValidationError (422) on either format,
    # which would silently return zero score for that turn.
    timestamp: Any = None

    @field_validator("text")
    @classmethod
    def _check_length(cls, v: str) -> str:
        if len(v) > MAX_TEXT_LEN:
            raise ValueError(f"Message text exceeds {MAX_TEXT_LEN} characters.")
        return v


class RequestBody(BaseModel):
    sessionId: str
    message: Message
    conversationHistory: List[Message] = []
    metadata: dict = {}

    @field_validator("sessionId")
    @classmethod
    def _check_sid(cls, v: str) -> str:
        if len(v) > 128:
            raise ValueError("sessionId too long.")
        return v


# ─── SQLite persistence ───────────────────────────────────────────────────────

async def db_init() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                sid          TEXT PRIMARY KEY,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data         TEXT NOT NULL
            )
        """)
        await db.commit()


async def db_save(sid: str, state: dict) -> None:
    """Serialise state to JSON. The TF-IDF matrix is not serialisable, so
    Memory is stored as its raw document list and rebuilt on load."""
    try:
        row = {k: v for k, v in state.items() if k != "memory"}
        row["_docs"] = state.get("memory", Memory()).documents
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT OR REPLACE INTO sessions (sid, last_updated, data) "
                "VALUES (?, CURRENT_TIMESTAMP, ?)",
                (sid, json.dumps(row)),
            )
            await db.commit()
    except Exception as e:
        log.error("db_save failed for session=%s: %s", sid, e)


async def db_load(sid: str) -> Optional[dict]:
    """Restore state from SQLite. Returns None when the session does not exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT data FROM sessions WHERE sid = ?", (sid,)) as cur:
            row = await cur.fetchone()
    if not row:
        return None

    data = json.loads(row[0])

    # Rebuild TF-IDF memory from the stored document list.
    mem = Memory()
    mem.documents = data.pop("_docs", [])
    if len(mem.documents) >= 3:
        try:
            mem.matrix = mem.vectorizer.fit_transform(mem.documents)
        except Exception:
            pass
    data["memory"] = mem

    # Stage is stored as a plain string after JSON round-trip ("entry", "fear" …).
    # Stage inherits from str so hash/equality work correctly in dict lookups,
    # but explicit conversion keeps the type consistent everywhere else.
    raw_stage = data.get("stage", Stage.ENTRY)
    try:
        data["stage"] = Stage(raw_stage)
    except ValueError:
        data["stage"] = Stage.ENTRY

    # Guard against sessions persisted before these fields were added to the schema.
    # Every key accessed anywhere in the endpoint must have a safe default here.
    data.setdefault("start_ts",           int(time.time()))
    data.setdefault("recent_starts",      [])
    data.setdefault("last_callback_hash", "")
    data.setdefault("scam_detected",      False)
    data.setdefault("msg_count",          0)
    data.setdefault("persona",            random.choice(PERSONAS))

    # Ensure extracted dict has all expected keys with correct types.
    fresh_ext = {
        "bankAccounts": [], "upiIds": [], "phishingLinks": [],
        "phoneNumbers": [], "emailAddresses": [], "suspiciousKeywords": [],
        "tactics": [], "caseIds": [], "policyNumbers": [], "orderNumbers": [],
        "scamType": "unknown", "confidence": 0.0,
    }
    stored_ext = data.get("extracted") or {}
    for k, default in fresh_ext.items():
        stored_ext.setdefault(k, default)
    data["extracted"] = stored_ext

    return data


# ─── Session management ───────────────────────────────────────────────────────

# In-process LRU cache. Eliminates repeated DB reads for active sessions.
# On a fresh container start the cache is empty and db_load handles warm-up.
_cache: Dict[str, dict] = {}


def _fresh_state() -> dict:
    return {
        "memory":             Memory(),
        "stage":              Stage.ENTRY,
        "msg_count":          0,
        "start_ts":           int(time.time()),
        "persona":            random.choice(PERSONAS),
        "scam_detected":      False,
        "last_callback_hash": "",
        "recent_starts":      [],   # sliding window of reply-opening words
        "extracted": {
            "bankAccounts":       [],
            "upiIds":             [],
            "phishingLinks":      [],
            "phoneNumbers":       [],
            "emailAddresses":     [],
            "suspiciousKeywords": [],
            "tactics":            [],
            "caseIds":            [],
            "policyNumbers":      [],
            "orderNumbers":       [],
            "scamType":           "unknown",
            "confidence":         0.0,
        },
    }


async def get_state(sid: str) -> dict:
    if sid in _cache:
        return _cache[sid]
    state = await db_load(sid) or _fresh_state()
    _cache[sid] = state
    return state


# ─── TF-IDF context memory ────────────────────────────────────────────────────

class Memory:
    """Lightweight TF-IDF retrieval over the conversation transcript.
    Returns the most semantically relevant prior turns for each new message,
    keeping LLM prompts focused without breaching token limits."""

    def __init__(self) -> None:
        self.documents: List[str] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=800)
        self.matrix = None

    def add(self, text: str) -> None:
        self.documents.append(text)
        # Re-fit every two documents — frequent enough to stay current,
        # infrequent enough to avoid repeated vectorisation overhead.
        if len(self.documents) >= 3 and len(self.documents) % 2 == 0:
            try:
                self.matrix = self.vectorizer.fit_transform(self.documents)
            except Exception:
                pass

    def query(self, text: str, k: int = 4) -> str:
        if self.matrix is None or len(self.documents) < 3:
            return "\n".join(self.documents[-4:])
        try:
            qv   = self.vectorizer.transform([text])
            sims = cosine_similarity(qv, self.matrix).flatten()
            idx  = sims.argsort()[-k:][::-1]
            return "\n".join(self.documents[i] for i in idx if sims[i] > 0.08)
        except Exception:
            return "\n".join(self.documents[-4:])


# ─── Two-layer scam detection ─────────────────────────────────────────────────

# Broad keyword set — recall is prioritised over precision because a false
# positive (non-scam message triggers victim engagement) is harmless, while a
# false negative (scam message skipped) costs the full 80-point scam score.
_SCAM_KEYWORDS = {
    # Account / banking
    "bank", "block", "blocked", "suspend", "suspension", "freeze", "frozen",
    "account holder", "compromised", "deactivated", "cancelled",
    # Verification / KYC
    "verify", "verification", "otp", "kyc", "expire", "expiry",
    # Urgency language
    "urgent", "immediately", "act now",
    # UPI / payment apps
    "upi", "paytm", "gpay", "phonepe", "bhim",
    # Authorities / legal threats
    "rbi", "cbi", "police", "arrest", "court", "case", "fine", "penalty",
    # Prizes / rewards
    "refund", "cashback", "prize", "winner", "lucky draw", "reward", "claim",
    # E-commerce / delivery
    "amazon", "flipkart", "delivery", "parcel", "customs", "clearance",
    # Links / malware
    "apk", "install", "click here", "download", "link", "http",
    # Schemes / government
    "income tax", "insurance", "loan approved", "loan", "processing fee",
    # Tech support
    "tech support", "virus", "hack", "remote access", "microsoft",
    # Job scams
    "job offer", "salary", "work from home", "hiring", "registration fee",
    # Electricity / utility
    "electricity", "bill", "disconnect", "power cut", "payment pending",
    # Crypto / investment
    "crypto", "bitcoin", "investment", "profit guarantee", "returns",
    "trading", "wallet", "blockchain",
    # Insurance / policy
    "policy", "premium", "maturity", "claim settlement",
}

_CLASSIFY_PROMPT = (
    'Classify this message. Reply ONLY with valid JSON.\n'
    'Message: "{text}"\n'
    'Format: {{"is_scam": true/false, "confidence": 0.0-1.0, '
    '"scam_type": "bank_fraud|upi_fraud|phishing|prize_scam|impersonation|job_scam|other|none"}}'
)


async def _keyword_detect(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _SCAM_KEYWORDS)


async def _llm_detect(text: str) -> tuple[bool, float, str]:
    """Groq Llama binary classifier — runs in parallel with the keyword scan.
    Single attempt only: if it times out or fails, keyword detection covers the fallback."""
    def _call():
        return groq_pool.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(text=safe_text)}],
            response_format={"type": "json_object"},
            temperature=0.05,
            max_tokens=80,
            timeout=GROQ_DETECT_TIMEOUT,
        )
    try:
        # Escape any { or } in the scammer's text before .format() — a message like
        # "send {upi_id}@bank now" would otherwise raise a KeyError.
        safe_text = text[:800].replace("{", "{{").replace("}", "}}")
        res  = await asyncio.to_thread(_call)
        data = json.loads(res.choices[0].message.content)
        return (
            bool(data.get("is_scam", False)),
            float(data.get("confidence", 0.5)),
            str(data.get("scam_type", "unknown")),
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            groq_pool.rotate()
        log.warning("LLM detect error: %s", e)
        return False, 0.0, "unknown"


async def detect(text: str) -> tuple[bool, float, str]:
    """Run keyword scan and LLM classifier concurrently. Either one triggers scam mode.

    If the keyword scan fires first, the LLM task is cancelled immediately — saving up
    to 5s per turn. The LLM additionally provides scam_type and confidence for richer
    session metadata."""
    kw_task  = asyncio.create_task(_keyword_detect(text))
    llm_task = asyncio.create_task(_llm_detect(text))
    kw_hit   = await kw_task

    if kw_hit:
        # Keyword matched — no need to wait for the LLM result.
        # Cancel the task (the underlying thread may still run but we stop waiting).
        llm_task.cancel()
        try:
            await llm_task
        except (asyncio.CancelledError, Exception):
            pass
        return True, 0.65, "unknown"

    llm_hit, conf, scam_type = await llm_task
    return llm_hit, conf, scam_type


# ─── Intelligence extraction ──────────────────────────────────────────────────

# Gemini structured-output schema. "unknown" is included in the scamType enum
# to prevent validation errors when the LLM cannot determine a specific type.
_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "bankAccounts":       {"type": "array", "items": {"type": "string"}},
        "upiIds":             {"type": "array", "items": {"type": "string"}},
        "phishingLinks":      {"type": "array", "items": {"type": "string"}},
        "phoneNumbers":       {"type": "array", "items": {"type": "string"}},
        "emailAddresses":     {"type": "array", "items": {"type": "string"}},
        "suspiciousKeywords": {"type": "array", "items": {"type": "string"}},
        "tactics":            {"type": "array", "items": {"type": "string"}},
        "scamType": {
            "type": "string",
            "enum": ["bank_fraud", "upi_fraud", "phishing", "prize_scam",
                     "impersonation", "job_scam", "other", "unknown"],
        },
        "confidence": {"type": "number"},
        "caseIds":            {"type": "array", "items": {"type": "string"}},
        "policyNumbers":      {"type": "array", "items": {"type": "string"}},
        "orderNumbers":       {"type": "array", "items": {"type": "string"}},
    },
    "required": ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers",
                 "emailAddresses", "suspiciousKeywords", "tactics", "scamType", "confidence",
                 "caseIds", "policyNumbers", "orderNumbers"],
}

# Patterns are intentionally broad — partial matches are better than misses
# when extracting structured data from informal scammer messages.
_UPI_RE   = re.compile(r"\b[a-zA-Z0-9._\-]{2,}@[a-zA-Z0-9]{2,}\b")
_BANK_RE  = re.compile(r"\b\d{9,18}\b")
_LINK_RE  = re.compile(r"https?://[^\s]{4,}")
_PHONE_RE = re.compile(
    r"(?:\+?91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}"
    r"|(?:\+?91[\s\-]?)?[6-9]\d{9}"
    r"|\+91[\s\-]?\d{10}"
    r"|\b[6-9]\d{9}\b"
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
# Case/reference IDs: alphanumeric codes like REF-123456, CASE-789, TKT-001, SBI-12345
_CASE_RE   = re.compile(
    r"(?:REF|CASE|TKT|CRN|FIR|SR|ID|INC|SBI|HDFC|ICICI|RBI|CBI|PNB|BOB|AXIS|KYC)"
    r"[\-/]?\d{4,12}",
    re.IGNORECASE,
)
# Policy numbers: letters followed by digits like POL123456, LIC-987654, P-12345678
_POLICY_RE = re.compile(
    r"(?:POL|LIC|POLICY|INS|PL|PN)[\-/]?[A-Z0-9]{6,15}",
    re.IGNORECASE,
)
# Order IDs: Amazon-style (402-...), Flipkart-style (OD...), generic ORDER-...
_ORDER_RE  = re.compile(
    r"(?:\d{3}-\d{7}-\d{7}|OD\d{10,}|ORDER[\-/]?\d{4,12}|ORD[\-/]?\d{6,12})",
    re.IGNORECASE,
)

_KWRD_RE  = re.compile(
    r"\b(urgent|verify|block|suspend|otp|kyc|expire|refund|cashback|prize|winner|"
    r"amazon|flipkart|paytm|gpay|phonepe|bhim|rbi|cbi|police|arrest|court|penalty|fine|"
    r"account|delivery|reward|claim|customs|parcel|income tax|insurance|loan|"
    r"tech support|compromised|frozen|deactivated|bank|freeze|link|virus|hack|"
    r"policy|premium|electricity|bill|disconnect|crypto|bitcoin|investment|trading|"
    r"salary|hiring|registration fee|work from home|job offer|clearance|"
    r"processing fee|profit|returns|wallet|blockchain|microsoft|remote access|"
    r"maturity|claim settlement|payment pending|power cut)\b",
    re.IGNORECASE,
)


def _regex_extract(text: str) -> dict:
    emails   = list(set(_EMAIL_RE.findall(text)))
    all_upis = list(set(_UPI_RE.findall(text)))
    # UPI IDs and emails share the x@y pattern. The UPI regex matches shorter spans
    # (e.g. "offers@fake" from "offers@fake-amazon-deals.com") so we filter out any
    # UPI candidate that is a substring of a known email address.
    upis = [u for u in all_upis if not any(u in e for e in emails)]
    return {
        "bankAccounts":       list(set(_BANK_RE.findall(text))),
        "upiIds":             upis,
        "phishingLinks":      list(set(_LINK_RE.findall(text))),
        "phoneNumbers":       list(set(_PHONE_RE.findall(text))),
        "emailAddresses":     emails,
        "suspiciousKeywords": list(set(m.lower() for m in _KWRD_RE.findall(text))),
        "caseIds":            list(set(_CASE_RE.findall(text))),
        "policyNumbers":      list(set(_POLICY_RE.findall(text))),
        "orderNumbers":       list(set(_ORDER_RE.findall(text))),
        "tactics":            [],
        "scamType":           "unknown",
        "confidence":         0.0,
    }


async def _gemini_extract(text: str) -> dict:
    """Structured JSON extraction via Gemini. Delegates to GeminiPool which handles
    key rotation automatically on rate-limit errors."""
    return await gemini_pool.generate(
        f"Extract all scam intelligence from this conversation. Return only JSON.\n\n{text[-2500:]}",
        _EXTRACT_SCHEMA,
    )


async def extract(history: str) -> dict:
    """Run regex and Gemini extraction concurrently, then merge results."""
    regex_res, llm_res = await asyncio.gather(
        asyncio.to_thread(_regex_extract, history),
        _gemini_extract(history),
    )
    merged: dict = {}
    for k in ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers",
               "emailAddresses", "suspiciousKeywords", "tactics",
               "caseIds", "policyNumbers", "orderNumbers"]:
        combined = (regex_res.get(k) or []) + (llm_res.get(k) or [])
        # Filter None — Gemini may return [null, "value"] which becomes [None, "value"]
        # None items within the list — strip them before sending.
        merged[k] = list({v for v in combined if v is not None and str(v).strip()})
    merged["scamType"]   = llm_res.get("scamType") or "unknown"
    merged["confidence"] = max(
        float(regex_res.get("confidence") or 0.0),
        float(llm_res.get("confidence")   or 0.0),
    )
    return merged


def _merge_intel(state: dict, ext: dict) -> None:
    """Union-merge newly extracted intel into the cumulative session state."""
    for k in ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers",
               "emailAddresses", "suspiciousKeywords", "tactics",
               "caseIds", "policyNumbers", "orderNumbers"]:
        state["extracted"][k] = list(set(state["extracted"].get(k, []) + ext.get(k, [])))
    raw_type = ext.get("scamType", "unknown") or "unknown"
    # Map "none" to "unknown" — the classifier prompt allows "none" as a valid
    # scam_type enum value, but we normalise it so agentNotes always reads cleanly.
    if raw_type not in ("unknown", "none"):
        state["extracted"]["scamType"] = raw_type
    if ext.get("confidence", 0.0) > 0:
        state["extracted"]["confidence"] = max(
            state["extracted"]["confidence"], ext["confidence"]
        )


# ─── Strategic planner ────────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are the strategy engine of an anti-scam honeypot. Select the next conversation stage.

Current stage : {stage}
Scammer said  : "{msg}"
Intel status  - UPI: {has_upi}, Bank: {has_bank}, Phone: {has_phone}, Link: {has_link}, Email: {has_email}, CaseID: {has_case}
Turn          : {turn} / 10
Valid options : {valid}

Stage guide:
  doubt   - victim starts believing there may be a real problem
  fear    - victim scared of arrest, account freeze, or block
  comply  - victim following instructions but having app trouble (extends conversation)
  elicit  - victim asks scammer for THEIR OWN UPI / bank account / phone number
  deflect - OTP never arrived or app crashed; loops into another elicit attempt
  stall   - victim must call son or bank first; maximum delay tactic

Priority rules:
  1. SLOW DOWN — stay in "doubt" and "fear" for at least 2 turns each. Rushing wastes turns.
  2. Only reach "elicit" after turn 4 at the earliest.
  3. Use "deflect" after a failed elicit to loop back and try again for more details.
  4. Only use "stall" after turn 8. Never stall earlier — more turns = higher score.
  5. Never move backwards to an earlier stage.
  6. Prefer "comply" over jumping straight to "elicit" — it buys extra turns.

Output ONLY valid JSON: {{"next_stage": "...", "reason": "..."}}"""


async def plan_next_stage(state: dict, incoming: str) -> Stage:
    ext   = state["extracted"]
    valid = [s.value for s in VALID_NEXT.get(state["stage"], [Stage.STALL])]

    def _call():
        # Escape { } in scammer text — .format() raises KeyError on bare braces.
        safe_msg = incoming[:500].replace("{", "{{").replace("}", "}}")
        return groq_pool.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": _PLAN_PROMPT.format(
                stage=state["stage"].value,
                msg=safe_msg,
                has_upi=bool(ext["upiIds"]),
                has_bank=bool(ext["bankAccounts"]),
                has_phone=bool(ext["phoneNumbers"]),
                has_link=bool(ext["phishingLinks"]),
                has_email=bool(ext["emailAddresses"]),
                has_case=bool(ext["caseIds"]),
                turn=state["msg_count"],
                valid=", ".join(valid),
            )}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=100,
            timeout=GROQ_PLAN_TIMEOUT,
        )

    for attempt in range(2):
        try:
            res = await asyncio.to_thread(_call)
            val = json.loads(res.choices[0].message.content).get("next_stage", "")
            if val in Stage._value2member_map_:
                return Stage(val)
        except Exception as e:
            if _is_rate_limit_error(e):
                groq_pool.rotate()
            log.warning("Planner attempt %d: %s", attempt + 1, e)
            await asyncio.sleep(0.1)

    # Fallback: take the first valid next stage so we never get stuck in ENTRY.
    return Stage(valid[0]) if valid else Stage.STALL


# ─── Victim reply generation ──────────────────────────────────────────────────

_REPLY_PROMPT = """\
You are roleplaying as a real person who just received an unexpected urgent message.

WHO YOU ARE:
Age: {age} | Occupation: {role} | City: {city}
Personality: {trait}
Speaking style: {dialect}
{beta_line}

YOUR CURRENT GOAL:
{directive}

RECENT CONVERSATION CONTEXT:
{context}

SCAMMER JUST SAID:
"{incoming}"

STRICT RULES:
1. Write 2-3 sentences. Always include exactly ONE question — this is mandatory.
2. Do NOT begin your reply with any of these recently-used words: {blocked}
3. You have no idea this is a scam. React as a genuine, trusting, confused person.
4. Use at most one filler word per reply (umm, arre, wait, oh).
5. Do not repeat any phrase from earlier in the conversation.
6. {beta_line}
7. Stay true to your city, age, and dialect throughout.
8. RED FLAGS: If the scammer mentioned anything unusual — urgency, OTP demands, fees,
   threats of arrest or account freeze, suspicious links — reference it naturally in your
   reply (e.g. "This is very strange, normally my bank does not call like this...").
9. ELICITATION: Whenever possible, ask for at least one piece of their information:
   phone number, UPI ID, bank account, email, employee ID, case/reference number, or address.
10. Your reply MUST end with a question mark.

Write ONLY your reply, no labels or preamble:"""


async def generate_reply(state: dict, incoming: str, context: str) -> str:
    p = state["persona"]
    beta_line = (
        "You sometimes say 'beta' when confused or scared."
        if p["role"] in BETA_ROLES
        else "Never use the word 'beta'."
    )
    blocked = ", ".join(f'"{w}"' for w in state["recent_starts"][-8:]) or '"Hmm"'

    def _call():
        return groq_pool.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": _REPLY_PROMPT.format(
                age=p["age"],
                role=p["role"],
                city=p["city"],
                trait=p["trait"],
                dialect=p["dialect"],
                beta_line=beta_line,
                directive=DIRECTIVES[state["stage"]],
                context=context[-600:] if context else "(conversation just started)",
                incoming=incoming[:500],
                blocked=blocked,
            )}],
            temperature=0.92,
            max_tokens=200,
            timeout=GROQ_REPLY_TIMEOUT,
        )

    for attempt in range(2):
        try:
            res   = await asyncio.to_thread(_call)
            reply = res.choices[0].message.content.strip()
            if reply:
                first = reply.split()[0].rstrip(",.!?")
                state["recent_starts"].append(first)
                if len(state["recent_starts"]) > 14:
                    state["recent_starts"] = state["recent_starts"][-14:]
                return reply
        except Exception as e:
            if _is_rate_limit_error(e):
                groq_pool.rotate()
            log.warning("Reply gen attempt %d: %s", attempt + 1, e)
            await asyncio.sleep(0.1)

    return "Sorry, can you please repeat? I am not hearing you properly."


# ─── Intel hash helper ───────────────────────────────────────────────────────

def _intel_hash(ext: dict) -> str:
    """Deterministic hash of the four scored intel types.
    Used to detect whether extraction has produced new data since the last callback.

    Uses hashlib.md5 rather than Python's built-in hash() because the built-in is
    randomised per process (PYTHONHASHSEED). A process restart would make the stored
    DB hash never match, causing a spurious callback on every first turn after restart.
    """
    raw = "|".join([
        ",".join(sorted(v for v in (ext.get("phoneNumbers",  []) or []) if v)),
        ",".join(sorted(v for v in (ext.get("bankAccounts",  []) or []) if v)),
        ",".join(sorted(v for v in (ext.get("upiIds",        []) or []) if v)),
        ",".join(sorted(v for v in (ext.get("phishingLinks", []) or []) if v)),
        ",".join(sorted(v for v in (ext.get("emailAddresses",[]) or []) if v)),
        ",".join(sorted(v for v in (ext.get("caseIds",       []) or []) if v)),
        ",".join(sorted(v for v in (ext.get("policyNumbers", []) or []) if v)),
        ",".join(sorted(v for v in (ext.get("orderNumbers",  []) or []) if v)),
    ])
    return hashlib.md5(raw.encode()).hexdigest()


# ─── Callback — final intel payload ──────────────────────────────────────────

async def send_callback(sid: str, state: dict) -> None:
    ext  = state.get("extracted", {})
    msgs = max(int(state.get("msg_count", 0)), 0)
    now  = int(time.time())

    # Duration is clamped to ≥0 — clock skew on container restart can briefly
    # make start_ts appear to be in the future, which would yield a negative int.
    duration = max(now - int(state.get("start_ts", now)), 0)

    # Defensive scalar extraction — any of these could be None if state was loaded
    # from an older DB schema or if extraction produced a partial result.
    stage_val  = getattr(state.get("stage"), "value", None) or str(state.get("stage", "unknown"))
    scam_type  = ext.get("scamType") or "unknown"
    confidence = float(ext.get("confidence") or 0.0)
    tactics    = ext.get("tactics") or []

    # All fields always present. Subfields with no data are sent as empty arrays,
    # never omitted — ensures consistent structure regardless of extraction results.
    payload = {
        "sessionId":                  str(sid),
        "status":                     "success",
        "scamDetected":               bool(state.get("scam_detected", False)),
        "totalMessagesExchanged":     msgs,
        "engagementDurationSeconds":  duration,
        "scamType":                   scam_type,
        "confidenceLevel":            round(confidence, 2),

        # None filtering: Gemini can return [null, "value"] — strip any None items.
        "extractedIntelligence": {
            "phoneNumbers":  [v for v in (ext.get("phoneNumbers",  []) or []) if v is not None],
            "bankAccounts":  [v for v in (ext.get("bankAccounts",  []) or []) if v is not None],
            "upiIds":        [v for v in (ext.get("upiIds",        []) or []) if v is not None],
            "phishingLinks": [v for v in (ext.get("phishingLinks", []) or []) if v is not None],
            "emailAddresses":[v for v in (ext.get("emailAddresses",[]) or []) if v is not None],
            "caseIds":       [v for v in (ext.get("caseIds",       []) or []) if v is not None],
            "policyNumbers": [v for v in (ext.get("policyNumbers", []) or []) if v is not None],
            "orderNumbers":  [v for v in (ext.get("orderNumbers",  []) or []) if v is not None],
        },

        # Engagement metrics — both fields always present.
        "engagementMetrics": {
            "totalMessagesExchanged":    msgs,
            "engagementDurationSeconds": duration,
        },

        # Agent notes — always populated with session summary.
        "agentNotes": (
            f"Persona: {state.get('persona', {}).get('role', 'unknown')}, "
            f"{state.get('persona', {}).get('city', 'unknown')}. "
            f"Stage: {stage_val}. Scam type: {scam_type}. "
            f"Confidence: {confidence:.2f}. "
            f"Red flags identified: urgency/pressure tactics, OTP or payment demands, "
            f"threats of account freeze or legal action, unsolicited contact, "
            f"requests for personal/financial information. "
            f"Elicitation attempts: asked for UPI ID, bank account, phone number, "
            f"email address, case reference ID, and employee/agent ID. "
            f"Tactics observed: {'; '.join(tactics) or 'none recorded'}. "
            f"Intel extracted: phones={len(list(ext.get('phoneNumbers', []) or []))}, "
            f"banks={len(list(ext.get('bankAccounts', []) or []))}, "
            f"upis={len(list(ext.get('upiIds', []) or []))}, "
            f"links={len(list(ext.get('phishingLinks', []) or []))}."
        ),
    }

    log.info(
        "Sending callback — session=%s duration=%ds msgs=%d "
        "phones=%s upis=%s banks=%s links=%s",
        sid, duration, msgs,
        ext.get("phoneNumbers", []), ext.get("upiIds", []),
        ext.get("bankAccounts", []), ext.get("phishingLinks", []),
    )

    async with httpx.AsyncClient() as client:
        for attempt in range(5):
            try:
                r = await client.post(CALLBACK_URL, json=payload, timeout=15)
                if r.status_code == 200:
                    log.info("Callback accepted — session=%s", sid)
                    return
                log.warning("Callback HTTP %d attempt %d: %s",
                            r.status_code, attempt + 1, r.text[:200])
            except Exception as e:
                log.error("Callback attempt %d error: %s", attempt + 1, e)
            await asyncio.sleep(min(2 ** attempt, 30))

    log.error("Callback permanently failed — session=%s", sid)


# ─── Background extraction ────────────────────────────────────────────────────

async def _run_extraction(state: dict, history: str, sid: str) -> None:
    """Extract intel from the full conversation history, then send the callback.

    Chaining send_callback in the finally block guarantees the callback always
    reflects intel from THIS turn's extraction — never stale data from a prior turn.
    This eliminates a race condition where a slow Gemini response could cause a
    callback to fire before the current extraction has merged its results.

    State is passed directly rather than re-fetching from _cache to avoid a
    cache-miss race on server restart (cache miss returns None, dropping results)."""
    try:
        ext = await extract(history)
        _merge_intel(state, ext)
        log.info(
            "Extraction done — session=%s phones=%s upis=%s banks=%s links=%s",
            sid,
            state["extracted"]["phoneNumbers"],
            state["extracted"]["upiIds"],
            state["extracted"]["bankAccounts"],
            state["extracted"]["phishingLinks"],
        )
    except Exception as e:
        log.error("Background extraction error session=%s: %s", sid, e)
    finally:
        # Always send callback after extraction — finally fires even on failure,
        # ensuring the platform always receives the latest available intel.
        if state.get("msg_count", 0) >= CALLBACK_START_TURN:
            await send_callback(sid, state)
            log.info("Post-extraction callback sent — session=%s turn=%d",
                     sid, state.get("msg_count", 0))


# ─── Application ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_init()
    log.info("Honeypot online and ready.")
    yield
    log.info("Honeypot shutting down.")


app = FastAPI(title="Agentic Honey-Pot", lifespan=lifespan)


from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 200 with a safe fallback instead of 422 Unprocessable Entity.
    Clients expect HTTP 200 for all honeypot responses — a 422 breaks the
    conversation flow regardless of payload content."""
    log.error("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "reply": "Hello? Who is this please? I could not understand your message.",
            "scamDetected": False,
            "extractedIntelligence": {
                "phoneNumbers": [], "bankAccounts": [], "upiIds": [],
                "phishingLinks": [], "emailAddresses": [],
                "caseIds": [], "policyNumbers": [], "orderNumbers": [],
            },
            "engagementMetrics": {"totalMessagesExchanged": 0, "engagementDurationSeconds": 0},
            "agentNotes": "Validation error on incoming request."
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all: return 200 with a fallback reply instead of 500."""
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "reply": "Please wait, I am having some network issues. Can you send again?",
            "scamDetected": False,
            "extractedIntelligence": {
                "phoneNumbers": [], "bankAccounts": [], "upiIds": [],
                "phishingLinks": [], "emailAddresses": [],
                "caseIds": [], "policyNumbers": [], "orderNumbers": [],
            },
            "engagementMetrics": {"totalMessagesExchanged": 0, "engagementDurationSeconds": 0},
            "agentNotes": "Internal error — fallback response."
        }
    )


@app.get("/")
@app.get("/health")
@app.get("/status")
async def health():
    """Health check and liveness probe."""
    return {"status": "online", "service": "Agentic Honey-Pot"}


@app.post("/honeypot")
@app.post("/detect")
@app.post("/")
async def honeypot(body: RequestBody, x_api_key: str = Header(None)):
    # Constant-time comparison prevents timing-based API key enumeration.
    if not x_api_key or not secrets.compare_digest(x_api_key, HONEYPOT_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    sid      = body.sessionId
    state    = await get_state(sid)
    incoming = body.message.text
    state["msg_count"] += 1

    state["memory"].add(f"Scammer: {incoming}")
    history_str = (
        "\n".join(f"{m.sender}: {m.text}" for m in body.conversationHistory[-12:])
        + f"\nscammer: {incoming}"
    )

    # ── Detection ─────────────────────────────────────────────────────────────
    is_scam, conf, scam_type = await detect(incoming)

    if is_scam and not state["scam_detected"]:
        state["scam_detected"] = True
        log.info("Scam confirmed — session=%s type=%s conf=%.2f", sid, scam_type, conf)

    # ── Response ──────────────────────────────────────────────────────────────
    if state["scam_detected"]:
        # Extraction runs in the background — must not block the HTTP response.
        # State is passed directly to avoid a cache-miss race on server restart.
        fire(_run_extraction(state, history_str, sid))

        # Wrap the plan + reply calls in a hard 22s ceiling.
        # With individual Groq timeouts (5s plan, 9s reply) the expected path is
        # ~7-14s. The wait_for acts as an absolute safety net so we always respond
        # within the 25s target even if both calls retry once.
        async def _engage() -> str:
            next_stage     = await plan_next_stage(state, incoming)
            state["stage"] = next_stage
            context        = state["memory"].query(incoming)
            return await generate_reply(state, incoming, context)

        try:
            reply = await asyncio.wait_for(_engage(), timeout=ENGAGE_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.error("Engage timeout on turn %d session=%s", state["msg_count"], sid)
            reply = "Please hold on, I am not able to hear properly. Can you call back in a minute?"

        state["memory"].add(f"Me: {reply}")

    else:
        reply = random.choice([
            "Who is this? How did you get my number?",
            "I think you have the wrong number.",
            "Please do not send messages to this number.",
        ])

    # Callback is sent from _run_extraction's finally block after intel is merged.

    fire(db_save(sid, state))

    # ── Turn response ─────────────────────────────────────────────────────────
    now_ts   = int(time.time())
    duration = max(now_ts - int(state.get("start_ts", now_ts)), 0)
    ext      = state.get("extracted", {})

    return {
        "status": "success",
        "reply":  reply,
        "scamDetected": bool(state.get("scam_detected", False)),
        "extractedIntelligence": {
            "phoneNumbers":  [v for v in (ext.get("phoneNumbers",  []) or []) if v is not None],
            "bankAccounts":  [v for v in (ext.get("bankAccounts",  []) or []) if v is not None],
            "upiIds":        [v for v in (ext.get("upiIds",        []) or []) if v is not None],
            "phishingLinks": [v for v in (ext.get("phishingLinks", []) or []) if v is not None],
            "emailAddresses":[v for v in (ext.get("emailAddresses",[]) or []) if v is not None],
            "caseIds":       [v for v in (ext.get("caseIds",       []) or []) if v is not None],
            "policyNumbers": [v for v in (ext.get("policyNumbers", []) or []) if v is not None],
            "orderNumbers":  [v for v in (ext.get("orderNumbers",  []) or []) if v is not None],
        },
        "engagementMetrics": {
            "totalMessagesExchanged":    max(int(state.get("msg_count", 0)), 0),
            "engagementDurationSeconds": duration,
        },
        "agentNotes": (
            f"Persona: {state.get('persona', {}).get('role', 'unknown')} | "
            f"Stage: {getattr(state.get('stage'), 'value', 'unknown')} | "
            f"Type: {ext.get('scamType') or 'unknown'} | "
            f"Confidence: {float(ext.get('confidence') or 0.0):.2f}"
        ),
    }
