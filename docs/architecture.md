# Architecture Overview

## High-Level Design
Agentic Honey-Pot follows a **Neuro-Symbolic** architecture combining rule-based detection with LLM reasoning.

### Core Components
- **Fast Detector**: Regex + keyword matching (instant scam detection)
- **Extractor Agent**: Gemini 2.5 Flash for structured intelligence extraction
- **Planner Agent**: Llama-3.3-70B for dynamic strategy selection
- **Victim Persona Engine**: Context-aware natural language generation
- **Memory System**: TF-IDF vector store for multi-turn context
- **Persistence**: SQLite for session history and recovery

## Data Flow
1. Scammer message → Fast Detector
2. If scam → Parallel: Planner + Extractor
3. Planner chooses strategy → Victim generates reply
4. Extractor pulls intel → Stored in session
5. Callback sent when sufficient intel gathered

## Why This Design?
- **Speed**: FastAPI + Groq = <800ms responses
- **Realism**: Dynamic personas + artificial imperfection
- **Accuracy**: Hybrid regex + LLM extraction
- **Scalability**: Modular agents, easy to add voice (Whisper) or reporting webhooks
