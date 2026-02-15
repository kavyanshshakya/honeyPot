# 🛡️ Agentic Honey-Pot

**Autonomous AI Honeypot for Scam Detection & Intelligence Extraction**

## Description
Agentic Honey-Pot is an autonomous AI system designed to actively engage digital scammers in natural, multi-turn conversations. Instead of blocking numbers, it mimics believable Indian victim personas to waste scammer time and extract forensic intelligence (bank accounts, UPI IDs, phishing links, phone numbers, and emails) in real-time.

The system was built for the **India AI Impact Buildathon 2026** by HCL GUVI.

## Tech Stack
- **Backend**: FastAPI (Python)
- **LLMs**: 
  - Groq Llama-3.3-70B (for natural persona simulation and conversation)
  - Google Gemini 2.5 Flash (for structured intelligence extraction)
- **Memory**: TF-IDF Vectorizer for context retrieval across turns
- **Persistence**: Async SQLite (`aiosqlite`) for session history
- **Deployment**: Render

## Approach

### Scam Detection
- Hybrid system: Fast regex + keyword matching for instant detection
- LLM confirmation (Gemini) for complex or ambiguous messages

### Intelligence Extraction
- Dual-layer extraction: Regex for immediate patterns + Gemini structured output for accuracy
- Supports: Bank accounts, UPI IDs, phone numbers, phishing links, email addresses
- Validates formats (e.g., +91 phone numbers, valid UPI patterns)

### Engagement Strategy
- Dynamic persona selection (10 realistic Indian victim profiles)
- Neuro-Symbolic planner (Llama-3.3) that chooses optimal tactics: feign_failure, bait_greed, stall, etc.
- Artificial imperfection (typos, latency simulation, varied openings) to avoid bot detection

### Persistence & Multi-turn
- Full conversation history stored in SQLite
- Context memory using TF-IDF for long conversations

## Scalability & Future Scope
While the current version is text-based, the architecture is modular and designed for easy extension. It is ready for:
- **Whisper API integration** for handling voice notes and regional dialect audio scams
- **Automated Reporting** to national cybercrime portals (e.g., via webhooks to cybercrime.gov.in)
- Multi-channel support (WhatsApp, SMS, Voice Calls)

## Setup Instructions
1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run locally: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`

## API Endpoint
- **URL**: `/honeypot`
- **Method**: `POST`
- **Authentication**: `x-api-key` header

## Deployment
Deployed on Render (https://guvi-honeypot-74ml.onrender.com).
