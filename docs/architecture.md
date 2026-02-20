# Architecture Overview

## Design Philosophy

Agentic Honey-Pot uses a **Neuro-Symbolic** architecture: fast rule based systems handle the common case (instant keyword detection, regex extraction), while LLMs handle the nuanced work (natural conversation, structured entity extraction). This keeps response times within the platform's 30-second limit while maximising intelligence quality.

---

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Endpoint                         │
│                    POST /honeypot  /detect  /                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Two-Layer      │
                    │  Scam Detector  │
                    └────────┬────────┘
              ┌──────────────┴──────────────┐
              │ Layer 1: Keyword Set        │ Layer 2: Groq Llama
              │ (instant, <1ms)             │ classifier (parallel,
              │ 60+ scam terms              │ 5s timeout)
              └──────────────┬──────────────┘
                             │ scam confirmed (sticky)
               ┌─────────────┴──────────────────────┐
               │ Background (fire-and-forget)        │ Blocking (~14s)
               │                                     │
       ┌───────▼────────┐                   ┌────────▼────────────┐
       │  Dual-Layer    │                   │  Stage Planner      │
       │  Extractor     │                   │  (Groq Llama, 5s)   │
       └───────┬────────┘                   └────────┬────────────┘
               │                                     │
       ┌───────▼────────┐                   ┌────────▼────────────┐
       │ Regex (instant)│                   │  Persona Reply Gen  │
       │ + Gemini 2.5   │                   │  (Groq Llama, 9s)   │
       │   (3–5s)       │                   └────────┬────────────┘
       └───────┬────────┘                            │
               │ intel merged into state             │ reply text
               │                                     │
       ┌───────▼────────┐                   ┌────────▼────────────┐
       │ send_callback  │                   │   HTTP Response     │
       │ (GUVI endpoint)│                   │   (200, ~14s total) │
       └───────┬────────┘                   └─────────────────────┘
               │
       ┌───────▼────────┐
       │  SQLite DB     │
       │  (db_save)     │
       └────────────────┘
```

---

## Component Deep-Dive

### 1. Two-Layer Scam Detector

**Layer 1 — Keyword Scan** (`_keyword_detect`)
- Pure Python set lookup: `any(kw in text.lower() for kw in _SCAM_KEYWORDS)`
- 60+ terms across 15 scam categories: bank fraud, UPI, KYC, phishing, prize, parcel, job, electricity, crypto, tech support, income tax, loan, insurance, refund, legal threat
- Returns in <1ms — runs as an `asyncio.Task` so Layer 2 can be cancelled immediately on a hit

**Layer 2 — LLM Classifier** (`_llm_detect`)
- Groq Llama-3.3-70B with JSON mode, 80 tokens, 5s timeout
- Returns `{is_scam, confidence, scam_type}`
- Catches subtle scams without keyword matches
- Cancelled immediately if Layer 1 fires (saves 5s per turn)

**Sticky Flag**: Once `scam_detected = True`, it never resets. `detect()` is called every turn, so a slow first message that misses detection is caught on turn 2.

---

### 2. Stage Machine (Planner)

A directed graph of 7 victim emotional states:

```
entry → doubt → fear → comply → elicit ↔ deflect → stall
```

| Stage | Victim Behaviour | Scoring Target |
|---|---|---|
| `entry` | Confused, asks who's calling | Red flag identification |
| `doubt` | Mentions red flags, asks for employee ID | Relevant questions |
| `fear` | Scared but asks for case number/helpline | Questions asked |
| `comply` | App "won't open", asks step-by-step | Turn count, elicitation |
| `elicit` | Asks for UPI/bank/email to "pay directly" | Intel extraction |
| `deflect` | OTP failed, asks for alternative account | Second elicitation attempt |
| `stall` | Needs son/daughter — asks for their email | Turn count, engagement |

**Planner rules**: Stay in `doubt`/`fear` for ≥2 turns each. Only reach `elicit` after turn 4. Only `stall` after turn 8 or when all intel types collected.

---

### 3. Dual-Layer Extractor

**Regex** (`_regex_extract`) — runs in a thread (non-blocking):
- `_PHONE_RE`: Indian mobile numbers in all formats (+91, 0, 10-digit, with dashes/spaces)
- `_UPI_RE`: `handle@provider` format, deduplicated against emails
- `_BANK_RE`: 9–18 digit sequences
- `_LINK_RE`: `http://` and `https://` URLs
- `_EMAIL_RE`: Standard RFC-ish email pattern
- `_CASE_RE`: Case/reference IDs (REF-, SBI-, CASE-, TKT-, CRN-)
- `_POLICY_RE`: Policy numbers (POL-, LIC-, INS-)
- `_ORDER_RE`: Order IDs (Amazon 402-xxx, Flipkart OD-, ORDER-)

**Gemini Structured Output** (`_gemini_extract`):
- Gemini 2.5 Flash with `response_mime_type: application/json` and a strict schema
- Returns all 8 intel types + `scamType` enum + `confidence` float
- Results union-merged with regex output — never overwrites, only accumulates

**Key safety**: All `None` values filtered before payload serialisation. Lists that are `None` (missing field) treated as `[]`.

---

### 4. Persona Engine

8 distinct victim personas (ages 55–72), each with:
- `age`, `role`, `city`: Sets realistic demographic expectations
- `trait`: Drives behavioural consistency (trusts authority, confuses easily, etc.)
- `dialect`: Natural language style (Hindi-English mix, formal English, regional phrases)

**Reply generation rules** (enforced in system prompt):
1. Every reply ends with a question mark (mandatory — scores Conversation Quality points)
2. Every reply mentions at least one red flag (urgency, OTP demand, freeze threat)
3. Every reply contains one elicitation attempt (asking for their contact details)
4. 1-word opening rotation prevents repetitive starts (`recent_starts` sliding window)

---

### 5. Callback Strategy

The GUVI callback is sent from `_run_extraction`'s `finally` block — **after** Gemini extraction completes and intel is merged into state. This means:

- Callback always contains the freshest possible intel from the current turn
- `totalMessagesExchanged` and `engagementDurationSeconds` are always live (computed at callback time)
- Fires even if extraction fails (Python `finally` always executes)
- The evaluation system's 10-second post-conversation wait window guarantees the final callback is received

---

### 6. Multi-Key Pool with Auto-Rotation

Both Groq and Gemini support multiple API keys loaded from `GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, etc.

On any 429/rate-limit/quota error, the pool rotates to the next key automatically. Each key is tried twice before giving up.

The Gemini pool uses a narrow `asyncio.Lock` (only around `genai.configure()` + model construction) so concurrent sessions don't block each other during the 3-5s network call.

---

### 7. Timing Budget

Per-turn worst-case (all calls retry once):

| Step | Time |
|---|---|
| Keyword detection | <1ms |
| Stage planning (Groq, 1 attempt) | ~2s |
| Reply generation (Groq, 1 attempt) | ~4s |
| Background: Gemini extraction | ~3-5s (non-blocking) |
| Hard ceiling (`asyncio.wait_for`) | 22s |
| Platform timeout | 30s |
| **Safety buffer** | **8s** |

If the 22s ceiling is hit, a graceful fallback reply is returned ("Please hold on, I cannot hear properly...") and the HTTP response is still 200.

---

### 8. Persistence & Recovery

Session state is stored in SQLite after every turn (`db_save` as a background task). On server restart:
- `db_load` rebuilds the full session state including TF-IDF memory corpus
- Stage enum is deserialized with a fallback to `ENTRY` if value is unrecognised
- All intel fields are scaffolded with defaults so old DB rows from earlier schema versions never cause `KeyError`

An in-process `_cache` dict avoids repeated DB reads for active sessions.
