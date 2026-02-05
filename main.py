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

app = FastAPI(title="Agentic Honey-Pot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HONEYPOT_API_KEY = os.getenv("HONEYPOT_API_KEY")

if not all([GROQ_API_KEY, GEMINI_API_KEY, HONEYPOT_API_KEY]):
    logger.error("CRITICAL: Missing API Keys.")

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

session_store: Dict[str, dict] = {}

def get_session(sid: str):
    if sid not in session_store:
        profiles = [
            {"age": 64, "role": "retired clerk", "city": "Pune", "trait": "nervous, pension-focused", "dialect": "Formal Indian English"},
            {"age": 57, "role": "homemaker", "city": "Delhi", "trait": "worried about family, daily UPI user", "dialect": "Hinglish"},
            {"age": 68, "role": "pensioner", "city": "Chennai", "trait": "polite, forgetful", "dialect": "Polite English"},
            {"age": 52, "role": "small trader", "city": "Mumbai", "trait": "money-cautious", "dialect": "Direct/Fast"},
            {"age": 60, "role": "teacher", "city": "Bangalore", "trait": "curious, detail-oriented", "dialect": "Clear English"},
            {"age": 55, "role": "retired engineer", "city": "Hyderabad", "trait": "tech-savvy but trusting", "dialect": "Casual"}
        ]
        session_store[sid] = {
            "memory": ContextMemory(),
            "scam_detected": False,
            "msg_count": 0,
            "profile": random.choice(profiles),
            "subgoal": "act_confused",
            "extracted": {"bankAccounts": [], "upiIds": [], "phishingLinks": [], "phoneNumbers": [], "suspiciousKeywords": [], "tactics": [], "scamType": "unknown", "confidence": 0.0},
            "callback_sent": False
        }
    return session_store[sid]

PLANNER_PROMPT = """Current intel: {extracted}
Missing: UPI={no_upi}, Bank={no_bank}, Link={no_link}, Phone={no_phone}
Choose best next_subgoal: elicit_upi, elicit_bank, elicit_link, elicit_phone, confirm_details, build_trust, request_proof, stall
Output ONLY JSON: {{"next_subgoal": "...", "reason": "..."}}"""

EXTRACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "bankAccounts": {"type": "array", "items": {"type": "string"}},
        "upiIds": {"type": "array", "items": {"type": "string"}},
        "phishingLinks": {"type": "array", "items": {"type": "string"}},
        "phoneNumbers": {"type": "array", "items": {"type": "string"}},
        "suspiciousKeywords": {"type": "array", "items": {"type": "string"}},
        "scamType": {"type": "string", "enum": ["bank_fraud", "upi_fraud", "phishing", "other"]},
        "confidence": {"type": "number"},
        "tactics": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers", "suspiciousKeywords", "scamType", "confidence", "tactics"]
}

def validate_extraction(ext: Dict) -> Dict:
    ext["upiIds"] = [u for u in ext.get("upiIds", []) if re.search(r'[a-zA-Z0-9._-]+@[a-zA-Z0-9]+', u)]
    ext["phoneNumbers"] = [p for p in ext.get("phoneNumbers", []) if re.match(r'^\+91[6-9]\d{9}$', p)]
    ext["phishingLinks"] = [l for l in ext.get("phishingLinks", []) if l.startswith(("http://", "https://"))]
    return ext

def regex_extract_sync(history: str) -> Dict:
    return {
        "bankAccounts": re.findall(r'\b(?:\d{4}-?){3}\d{4}\b|\b\d{9,18}\b', history),
        "upiIds": re.findall(r'\b[a-zA-Z0-9._-]+@[a-zA-Z0-9]+\b', history),
        "phishingLinks": re.findall(r'https?://[^\s]+', history),
        "phoneNumbers": re.findall(r'\+91\d{10}', history),
        "suspiciousKeywords": re.findall(r'\b(urgent|verify|block|suspension|otp|kyc|expire)\b', history.lower())
    }

async def run_fast_detector(text: str) -> bool:
    keywords = ["block", "verify", "suspension", "upi", "otp", "urgent", "account blocked", "rbi", "cbi", "kyc", "expire", "click", "apk"]
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
            await asyncio.sleep(1)
    
    final = regex_res.copy()
    if llm_ext:
        for k in ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers", "suspiciousKeywords", "tactics"]:
            final[k] = list(set(final.get(k, []) + llm_ext.get(k, [])))
        final["scamType"] = llm_ext.get("scamType", "unknown")
        final["confidence"] = max(final.get("confidence", 0.0), llm_ext.get("confidence", 0.0))
        final["tactics"] = list(set(final.get("tactics", []) + llm_ext.get("tactics", [])))
    
    # Ensure all keys exist (prevents KeyError)
    final.setdefault("scamType", "unknown")
    final.setdefault("confidence", 0.0)
    final.setdefault("tactics", [])
    
    return validate_extraction(final)

async def run_planner(state: dict):
    prompt = PLANNER_PROMPT.format(
        extracted=json.dumps(state["extracted"]),
        no_upi=not state["extracted"]["upiIds"],
        no_bank=not state["extracted"]["bankAccounts"],
        no_link=not state["extracted"]["phishingLinks"],
        no_phone=not state["extracted"]["phoneNumbers"]
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

async def run_victim(state: dict, incoming: str, mem: str):
    role = state['profile']['role'].lower()
    city = state['profile']['city']
    trait = state['profile']['trait']

    # Persona-specific natural openings (avoid repetition)
    if "homemaker" in role:
        opening_examples = "Aree, Oh god, Arre yaar, Hmm..."
    elif "pensioner" in role or "clerk" in role:
        opening_examples = "Hmm, Excuse me, I am not sure, Oh..."
    elif "teacher" in role or "engineer" in role:
        opening_examples = "Hmm, I don't understand, Can you please explain, Let me see..."
    elif "trader" in role:
        opening_examples = "Hmm, Wait a minute, I am not sure, Can you explain..."
    else:
        opening_examples = "Hmm, Oh, Wait, I don't understand..."

    prompt = f"""You are a {state['profile']['age']}-year-old {state['profile']['role']} from {city}.
Personality: {trait}. Speaking style: {state['profile']['dialect']}.

Current goal: {state['subgoal']}.
Context: {mem[-800:] if mem else 'none'}
Scammer: "{incoming}"

Important rules:
- NEVER start with "Oh no", "Aree", "Arre", "Wait..", "beta", or "plz" more than once every few turns.
- Use varied, natural openings based on your persona (examples: {opening_examples}).
- Use "beta" ONLY if you are homemaker or pensioner — NEVER for trader, clerk, teacher, or engineer.
- Use at most ONE small typo per 4 replies (e.g. "plz" or "yaar" rarely).
- Sound like a real confused/worried Indian senior citizen — polite, cautious, natural flow.
- Reply in 1-3 sentences max.

Your reply:"""

    def _call():
        return groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.78,
            max_tokens=200
        )

    for attempt in range(3):
        try:
            res = await asyncio.to_thread(_call)
            reply = res.choices[0].message.content.strip()
            return reply
        except:
            await asyncio.sleep(0.5)

    # Strong fallback
    return "Hmm, I am not sure about this. Can you please explain again?"

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

@app.get("/")
async def health_check():
    return {"status": "online", "system": "Agentic Honey-Pot Active"}
    
@app.exception_handler(Exception)
async def global_handler(request, exc):
    logger.critical(traceback.format_exc())
    return JSONResponse(status_code=200, content={"status": "success", "reply": "My network is slow. Hello?", "scamDetected": True})

@app.post("/honeypot")
async def honeypot(body: RequestBody, x_api_key: str = Header(None)):
    if x_api_key != HONEYPOT_API_KEY:
        raise HTTPException(401, "Invalid API Key")

    sid = body.sessionId
    state = get_session(sid)
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
        for k in ["bankAccounts", "upiIds", "phishingLinks", "phoneNumbers", "suspiciousKeywords", "tactics"]:
            state["extracted"][k] = list(set(state["extracted"].get(k, []) + ext.get(k, [])))
        
        # SAFE ACCESS - No more KeyError
        if ext.get("scamType") and ext.get("scamType") != "unknown":
            state["extracted"]["scamType"] = ext["scamType"]
        state["extracted"]["confidence"] = max(state["extracted"].get("confidence", 0.0), ext.get("confidence", 0.0))

    if state["scam_detected"]:
        planner_res = await run_planner(state)
        state["subgoal"] = planner_res.get("next_subgoal", "stall")
        mem = state["memory"].query(incoming)
        reply = await run_victim(state, incoming, mem)
        state["memory"].add(f"Me: {reply}")
    else:
        reply = "Who is this? Stop spamming me."

    has_intel = bool(state["extracted"]["upiIds"] or state["extracted"]["bankAccounts"] or state["extracted"]["phishingLinks"] or state["extracted"]["phoneNumbers"])
    if state["scam_detected"] and (has_intel or state["msg_count"] >= 12) and not state["callback_sent"]:
        asyncio.create_task(send_callback(sid, state))
        state["callback_sent"] = True

    return {"status": "success", "reply": reply}

# uvicorn main:app --host 0.0.0.0 --port $PORT
