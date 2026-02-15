import os
import json
import asyncio
import httpx
import logging
import random
import re
import traceback
from typing import Dict, List
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import aiosqlite

app = FastAPI(title="Agentic Honey-Pot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HONEYPOT_API_KEY = os.getenv("HONEYPOT_API_KEY")
DB_PATH = "honeypot.db"

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

class Message(BaseModel):
    sender: str
    text: str
    timestamp: int

class RequestBody(BaseModel):
    sessionId: str
    message: Message
    conversationHistory: List[Message] = []
    metadata: dict = {}

# ========================= SQLITE PERSISTENCE =========================
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                sid TEXT PRIMARY KEY,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data TEXT
            )
        """)
        await db.commit()

@app.on_event("startup")
async def startup():
    await init_db()

async def save_session(sid: str, state: dict):
    save_state = state.copy()
    save_state["memory"] = {"documents": state["memory"].documents}
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR REPLACE INTO sessions (sid, data) VALUES (?, ?)",
                         (sid, json.dumps(save_state)))
        await db.commit()

async def load_session(sid: str):
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT data FROM sessions WHERE sid = ?", (sid,)) as cursor:
            row = await cursor.fetchone()
            if row:
                data = json.loads(row[0])
                memory = ContextMemory()
                memory.documents = data.get("memory", {}).get("documents", [])
                if memory.documents:
                    try:
                        memory.matrix = memory.vectorizer.fit_transform(memory.documents)
                    except:
                        pass
                data["memory"] = memory
                return data
    return None

# ========================= SESSION MANAGEMENT =========================
session_store: Dict[str, dict] = {}

def get_session(sid: str):
    if sid not in session_store:
        profiles = [
            {"age": 22, "role": "final year student", "city": "Bangalore", "trait": "broke, desperate for a job, naive", "dialect": "Gen-Z slang, uses 'bro', 'sir is this real?', 'pls'"},
            {"age": 29, "role": "software engineer", "city": "Gurgaon", "trait": "busy, afraid of legal trouble/police", "dialect": "Professional English, short sentences, anxious"},
            {"age": 32, "role": "delivery partner", "city": "Mumbai", "trait": "needs quick cash, humble, very respectful", "dialect": "Hinglish, uses 'Sir ji', 'Bhaiya'"},
            {"age": 45, "role": "shop owner", "city": "Ahmedabad", "trait": "greedy, suspicious but wants high returns", "dialect": "Direct, broken English, talks about 'profit'"},
            {"age": 52, "role": "homemaker", "city": "Delhi", "trait": "excited about winning, chatting casually", "dialect": "Hindi-mixed English, enthusiastic"},
            {"age": 68, "role": "retired teacher", "city": "Chennai", "trait": "polite, slow, confused by OTPs", "dialect": "Formal English, apologetic, 'My son handles this'"},
            {"age": 58, "role": "government officer", "city": "Lucknow", "trait": "entitled, demands respect, slow to comply", "dialect": "Authoritative, Hindi-mixed"},
            {"age": 62, "role": "farmer", "city": "Punjab", "trait": "trusting, technology is magic/scary", "dialect": "Simple English, asks basic questions"},
            {"age": 48, "role": "working mother", "city": "Hyderabad", "trait": "harried, distracted, worried about kids", "dialect": "Hinglish, uses 'beta', 'bhaiya'"},
            {"age": 75, "role": "retired pensioner", "city": "Kolkata", "trait": "hard of hearing, forgets things, slow", "dialect": "Polite, formal, repeats questions"}
        ]
        session_store[sid] = {
            "memory": ContextMemory(),
            "history": [],
            "scam_detected": False,
            "msg_count": 0,
            "profile": random.choice(profiles),
            "subgoal": "act_confused",
            "extracted": {
                "bankAccounts": [],
                "upiIds": [],
                "phishingLinks": [],
                "phoneNumbers": [],
                "emailAddresses": [],
                "suspiciousKeywords": [],
                "tactics": [],
                "scamType": "unknown",
                "confidence": 0.0
            },
            "callback_sent": False,
            "recent_openings": []   # NEW: Tracks recent openings to avoid repetition
        }
    return session_store[sid]

# ========================= CONTEXT MEMORY =========================
class ContextMemory:
    def __init__(self):
        self.documents: List[str] = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=600)
        self.matrix = None

    def add(self, text: str):
        self.documents.append(text)
        if len(self.documents) >= 3 and len(self.documents) % 3 == 0:
            try:
                self.matrix = self.vectorizer.fit_transform(self.documents)
            except:
                pass

    def query(self, text: str, top_k: int = 3) -> str:
        if self.matrix is None or len(self.documents) < 3:
            return ""
        try:
            q_vec = self.vectorizer.transform([text])
            sims = cosine_similarity(q_vec, self.matrix).flatten()
            top_idx = sims.argsort()[-top_k:][::-1]
            relevant = [self.documents[i] for i in top_idx if sims[i] > 0.18]
            return "\n".join(relevant)
        except:
            return ""

# ========================= PROMPTS & SCHEMA =========================
PLANNER_PROMPT = """Current intel: {extracted}
Missing: UPI={no_upi}, Bank={no_bank}, Link={no_link}, Phone={no_phone}

STRATEGY MAP:
1. If they ask for money/OTP -> "feign_failure" (Pretend app crashed, ask for their details to try manual transfer).
2. If they threaten block -> "feign_panic" (Beg them to stop, offer to pay immediately).
3. If asking for details -> "bait_greed" (Mention having a high balance or being worried about a large transaction).
4. If missing specific intel (e.g., UPI) -> "elicit_missing" (Say 'My GPay is broken, give me YOUR UPI id').

Choose best next_subgoal: elicit_upi, elicit_bank, elicit_link, elicit_phone, feign_failure, feign_panic, bait_greed, stall
Output ONLY JSON: {{"next_subgoal": "...", "reason": "..."}}"""

EXTRACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "bankAccounts": {"type": "array", "items": {"type": "string"}},
        "upiIds": {"type": "array", "items": {"type": "string"}},
        "phishingLinks": {"type": "array", "items": {"type": "string"}},
        "phoneNumbers": {"type": "array", "items": {"type": "string"}},
        "emailAddresses": {"type": "array", "items": {"type": "string"}},
        "suspiciousKeywords": {"type": "array", "items": {"type": "string"}},
        "scamType": {"type": "string", "enum": ["bank_fraud", "upi_fraud", "phishing", "other"]},
        "confidence": {"type": "number"},
        "tactics": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers", "emailAddresses", "suspiciousKeywords", "scamType", "confidence", "tactics"]
}

def validate_extraction(ext: Dict) -> Dict:
    ext["upiIds"] = [u for u in ext.get("upiIds", []) if re.search(r'[a-zA-Z0-9._-]+@[a-zA-Z0-9]+', u)]
    ext["phoneNumbers"] = [p for p in ext.get("phoneNumbers", []) if re.search(r'(?:\+?91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|(?:\+?91[\-\s]?)?[6-9]\d{9}', p)]
    ext["phishingLinks"] = [l for l in ext.get("phishingLinks", []) if l.startswith(("http://", "https://"))]
    ext["emailAddresses"] = [e for e in ext.get("emailAddresses", []) if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', e)]
    return ext

def regex_extract_sync(history: str) -> Dict:
    return {
        "bankAccounts": re.findall(r'\b(?:\d{4}-?){3}\d{4}\b|\b\d{9,18}\b', history),
        "upiIds": re.findall(r'\b[a-zA-Z0-9._-]+@[a-zA-Z0-9]+\b', history),
        "phishingLinks": re.findall(r'https?://[^\s]+', history),
        "phoneNumbers": re.findall(r'(?:\+?91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|(?:\+?91[\-\s]?)?[6-9]\d{9}', history),
        "emailAddresses": re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', history),
        "suspiciousKeywords": re.findall(r'\b(urgent|verify|block|suspension|otp|kyc|expire|amazon|flipkart|paytm|delivery|order|refund|cashback|prize|winner)\b', history.lower())
    }

async def run_fast_detector(text: str) -> bool:
    keywords = ["block", "verify", "suspension", "upi", "otp", "urgent", "account blocked", "rbi", "cbi", "kyc", "expire", "click", "apk", "amazon", "flipkart", "paytm", "delivery", "order", "refund", "cashback", "prize", "winner"]
    return any(k in text.lower() for k in keywords)

async def run_extractor(history: str, latest: str) -> Dict:
    full = history + "\n" + latest
    regex_res = await asyncio.to_thread(regex_extract_sync, full)
    llm_ext = {}
    for attempt in range(3):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"response_mime_type": "application/json", "response_schema": EXTRACTOR_SCHEMA, "temperature": 0.1})
            prompt = f"Extract from:\n{full[-2000:]}\nOnly JSON."
            response = await model.generate_content_async(prompt)
            llm_ext = json.loads(response.text)
            break
        except Exception as e:
            logger.warning(f"Gemini fail {attempt}: {e}")
            await asyncio.sleep(0.5)
    final = regex_res.copy()
    if llm_ext:
        for k in ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers", "emailAddresses", "suspiciousKeywords", "tactics"]:
            final[k] = list(set(final.get(k, []) + llm_ext.get(k, [])))
        final["scamType"] = llm_ext.get("scamType", "unknown")
        final["confidence"] = max(final.get("confidence", 0.0), llm_ext.get("confidence", 0.0))
        final["tactics"] = list(set(final.get("tactics", []) + llm_ext.get("tactics", [])))
    final.setdefault("scamType", "unknown")
    final.setdefault("confidence", 0.0)
    final.setdefault("tactics", [])
    return validate_extraction(final)

async def run_planner(state: dict):
    prompt = PLANNER_PROMPT.format(
        extracted=json.dumps(state["extracted"]),
        no_upi=str(not bool(state["extracted"]["upiIds"])).lower(),
        no_bank=str(not bool(state["extracted"]["bankAccounts"])).lower(),
        no_link=str(not bool(state["extracted"]["phishingLinks"])).lower(),
        no_phone=str(not bool(state["extracted"]["phoneNumbers"])).lower()
    )
    def _call():
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=150
        )
    for attempt in range(3):
        try:
            res = await asyncio.to_thread(_call)
            return json.loads(res.choices[0].message.content)
        except:
            await asyncio.sleep(0.5)
    return {"next_subgoal": "stall"}

# ========================= IMPROVED VICTIM PROMPT (Fixed Repetition) =========================
async def run_victim(state: dict, incoming: str, mem: str):
    role = state['profile']['role'].lower()
    city = state['profile']['city']
    trait = state['profile']['trait']
    dialect = state['profile']['dialect']

    # Track recent openings to prevent repetition
    if "recent_openings" not in state:
        state["recent_openings"] = []

    prompt = f"""You are a real {state['profile']['age']}-year-old {state['profile']['role']} from {city}.
Personality: {trait}. Speaking style: {dialect}.

Current goal: {state['subgoal']}.
Recent conversation: {mem[-700:] if mem else 'start'}.
Scammer said: "{incoming}"

CRITICAL RULES (Follow strictly):
- NEVER repeat the same opening phrase. You have used these recently: {state["recent_openings"][-5:]} 
- Do NOT start with "Aree", "Arre", "Oh god", "Hmm...", "Wait..", "Oh no" more than once every 6-7 replies.
- Vary your starting words every single time. Be creative and natural.
- Use "beta" ONLY if you are homemaker or pensioner — NEVER for others.
- Use at most ONE small typo or hesitation per reply.
- Sound like a real confused/worried Indian senior citizen — polite, cautious, slightly slow.
- Reply in 1-3 short sentences max.

Generate a fresh, natural reply:"""

    def _call():
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.88,
            max_tokens=160
        )

    for attempt in range(3):
        try:
            res = await asyncio.to_thread(_call)
            reply = res.choices[0].message.content.strip()

            # Record the first word to avoid repetition next time
            first_word = reply.split()[0] if reply and reply.split() else "Hmm"
            state["recent_openings"].append(first_word)
            if len(state["recent_openings"]) > 10:
                state["recent_openings"] = state["recent_openings"][-10:]

            return reply
        except:
            await asyncio.sleep(0.5)

    return "I'm a bit confused... Can you say that again please?"

async def send_callback(sid: str, state: dict):
    payload = {
        "sessionId": sid,
        "scamDetected": True,
        "totalMessagesExchanged": state["msg_count"],
        "extractedIntelligence": state["extracted"],
        "agentNotes": f"Persona: {state['profile']['role']}. Subgoal: {state['subgoal']}. Type: {state['extracted']['scamType']}"
    }
    async with httpx.AsyncClient() as client:
        for attempt in range(4):
            try:
                r = await client.post("https://hackathon.guvi.in/api/updateHoneyPotFinalResult", json=payload, timeout=12)
                if r.status_code == 200:
                    logger.info(f"✅ Callback Success: {sid}")
                    return
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Callback Fail {attempt}: {e}")
                await asyncio.sleep(2 ** attempt)

# ========================= HEALTH CHECK =========================
@app.get("/")
async def health_check():
    return {"status": "online", "system": "Agentic Honey-Pot Active"}

# ========================= MAIN ENDPOINT =========================
@app.post("/honeypot")
async def honeypot(body: RequestBody, x_api_key: str = Header(None)):
    if x_api_key != HONEYPOT_API_KEY:
        raise HTTPException(401, "Invalid API Key")

    sid = body.sessionId
    state = await load_session(sid) or get_session(sid)
    state["msg_count"] += 1
    incoming = body.message.text

    state["memory"].add(f"Scammer: {incoming}")
    history_str = "\n".join([f"{m.sender}: {m.text}" for m in body.conversationHistory[-8:]]) + f"\nScammer: {incoming}"

    det_task = asyncio.create_task(run_fast_detector(incoming))
    ext_task = asyncio.create_task(run_extractor(history_str, incoming))

    is_scam = await det_task
    if is_scam:
        state["scam_detected"] = True

    ext = await ext_task
    if ext:
        for k in ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers", "emailAddresses", "suspiciousKeywords", "tactics"]:
            state["extracted"][k] = list(set(state["extracted"].get(k, []) + ext.get(k, [])))
        state["extracted"]["scamType"] = ext.get("scamType", "unknown")
        state["extracted"]["confidence"] = max(state["extracted"].get("confidence", 0.0), ext.get("confidence", 0.0))

    if state["scam_detected"]:
        planner_res = await run_planner(state)
        state["subgoal"] = planner_res.get("next_subgoal", "stall")
        mem = state["memory"].query(incoming)
        reply = await run_victim(state, incoming, mem)
        state["memory"].add(f"Me: {reply}")
    else:
        reply = "Who is this? Stop spamming me."

    has_intel = bool(state["extracted"]["upiIds"] or state["extracted"]["bankAccounts"] or state["extracted"]["phishingLinks"] or state["extracted"]["phoneNumbers"] or state["extracted"]["emailAddresses"])
    if state["scam_detected"] and (has_intel or state["msg_count"] >= 12) and not state["callback_sent"]:
        asyncio.create_task(send_callback(sid, state))
        state["callback_sent"] = True

    await save_session(sid, state)

    return {
        "status": "success",
        "reply": reply,
        "scamDetected": state["scam_detected"],
        "extractedIntelligence": state["extracted"],
        "engagementMetrics": {
            "totalMessagesExchanged": state["msg_count"],
            "engagementDurationSeconds": state["msg_count"] * 25
        },
        "agentNotes": f"Persona: {state['profile']['role']}. Subgoal: {state['subgoal']}. Type: {state['extracted']['scamType']}. Confidence: {state['extracted']['confidence']:.2f}"
    }

# uvicorn main:app --host 0.0.0.0 --port $PORT
