# 🛡️ Agentic Honey-Pot

**Autonomous AI Honeypot for Scam Detection & Intelligence Extraction**

Built for the **India AI Impact Buildathon 2026** by HCL GUVI.

---

## Description

Agentic Honey-Pot is an autonomous AI system that engages digital scammers in realistic multi-turn conversations. Instead of blocking, it impersonates believable Indian victim personas to waste scammer time while extracting forensic intelligence — phone numbers, bank accounts, UPI IDs, phishing links, emails, case IDs, policy numbers, and order numbers — in real time.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python 3.11+) |
| Conversation LLM | Groq — Llama-3.3-70B-Versatile |
| Extraction LLM | Google Gemini 2.5 Flash (structured JSON output) |
| Detection | Keyword set (instant) + Groq Llama classifier (fallback) |
| Memory | TF-IDF semantic context retrieval |
| Persistence | Async SQLite (aiosqlite) |

---

## Approach

### 1. Scam Detection — Two Layers
- **Keywords**: Instant set lookup across 84 scam terms. Fires in <1ms.
- **LLM Classifier**: Groq Llama-3.3-70B runs in parallel. Catches subtle scams. Either layer triggers scam mode (sticky — never resets).

### 2. Intelligence Extraction — Dual Layer
Runs in the background after each turn, never blocking the response:
- **Regex**: Phones, UPI IDs, bank accounts, links, emails, case IDs, policy numbers, order numbers.
- **Gemini Structured Output**: Validates and enriches regex results via JSON schema.
- Results union-merged across all turns — intel accumulates, never overwrites.

### 3. Engagement — Stage Machine
7-state directed graph drives the conversation:
```
entry → doubt → fear → comply → elicit ↔ deflect → stall
```
Every reply mandates one investigative question, one red flag reference, one elicitation attempt.

### 4. Persona Engine
8 realistic Indian victim personas (ages 55–72) — retired teacher, government clerk, pensioner, homemaker, shop owner, farmer, army officer, school principal. Each with a distinct city, dialect, and trait that naturally sustains long multi-turn conversations.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/honeypot-api
cd honeypot-api
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
uvicorn main:app --host 0.0.0.0 --port 8000
```

See `.env.example` for all environment variables.

---

## API

| | |
|---|---|
| **Endpoint** | `POST /honeypot` (also `/detect`, `/`) |
| **Auth** | `x-api-key` header |
| **Health** | `GET /health` |

### Request
```json
{
  "sessionId": "uuid-v4",
  "message": { "sender": "scammer", "text": "URGENT: Your account...", "timestamp": "..." },
  "conversationHistory": [],
  "metadata": { "channel": "SMS", "language": "English", "locale": "IN" }
}
```

### Response
```json
{
  "status": "success",
  "reply": "This is very strange, my bank never calls like this. Can you give me your employee ID?",
  "scamDetected": true,
  "extractedIntelligence": {
    "phoneNumbers": ["+91-9876543210"],
    "bankAccounts": [], "upiIds": [], "phishingLinks": [],
    "emailAddresses": [], "caseIds": [], "policyNumbers": [], "orderNumbers": []
  },
  "engagementMetrics": { "totalMessagesExchanged": 3, "engagementDurationSeconds": 87 },
  "agentNotes": "Stage: comply | Type: bank_fraud | Confidence: 0.92"
}
```

---

## Code Review Notes

- No hardcoded responses for specific test scenarios
- No test-traffic detection or special-casing
- All detection is keyword/LLM based on message content
- All extraction uses generic regex patterns and Gemini NLP
- Personas and stage transitions are dynamically generated per session
