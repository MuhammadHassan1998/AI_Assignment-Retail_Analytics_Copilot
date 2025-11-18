# Retail Analytics Copilot

This project is a friendly, local-first analytics partner for the Northwind dataset. It mixes relaxed document retrieval with grounded SQL so you can ask natural questions and still get verifiable answers—with citations every time.

## How the Agent Thinks

I split the brain into seven LangGraph nodes so each step stays predictable:

1. **Router (DSPy)** — decides whether a question needs docs, SQL, or both.
2. **Retriever** — TF-IDF over small markdown files with chunk-level scores.
3. **Planner** — pulls out dates, KPIs, and category hints before querying.
4. **NL→SQL (DSPy, optimized)** — generates SQLite that mirrors the live schema.
5. **Executor** — runs SQL, captures errors, and notes which tables were touched.
6. **Synthesizer (DSPy)** — formats the answer exactly as requested and attaches citations.
7. **Repair loop** — if SQL fails or the format is off, we retry (twice max) with extra guidance.

Everything is logged in the graph state so you can replay a decision path if something looks off.

### DSPy Optimization Snapshot

I tuned the NL→SQL module with BootstrapFewShot and a handful of representative prompts.

| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| Valid SQL rate | ~50% | ~75% | +25% |

The demos cover ranked revenue, marketing-window filters, and KPI math so the model sees the specific join patterns it needs.

### Practical Assumptions

- **Gross margin** falls back to `CostOfGoods ≈ 0.7 * UnitPrice` when the DB lacks cost data.
- **Retriever** sticks to TF-IDF because the corpus is tiny—no extra embeddings needed.
- **Synthesizer** tries to format results directly before touching the LLM, which saves time for simple scalar answers.
- **Repair loop** is capped at two attempts to keep runs deterministic.
- **Citations** only include doc chunks with a relevance score above 0.1 so the references stay helpful.

## Getting Started

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Ollama + model:**
   ```bash
   # Get Ollama from https://ollama.com
   ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
   ```
4. **Run Ollama locally:**
   ```bash
   ollama serve
   ```

## Running the Copilot

1. **Quick health check (optional but nice):**
   ```bash
   python test_setup.py
   ```
2. **Answer the evaluation questions:**
   ```bash
   python run_agent_hybrid.py \
     --batch sample_questions_hybrid_eval.jsonl \
     --out outputs_hybrid.jsonl
   ```

If you hit connection errors, double-check that Ollama is running and the Phi-3.5 model is downloaded.

## Output Contract

Each line in `outputs_hybrid.jsonl` follows:

```json
{
  "id": "question_id",
  "final_answer": <matches format_hint>,
  "sql": "<last executed SQL or empty>",
  "confidence": 0.0-1.0,
  "explanation": "<= 2 sentences>",
  "citations": ["table_names", "doc_chunk_ids"]
}
```

## Project Layout

```
assignment/
├── agent/
│   ├── graph_hybrid.py        # LangGraph agent
│   ├── dspy_signatures.py     # DSPy modules
│   ├── optimize.py            # NL→SQL tuning helper
│   ├── rag/
│   │   └── retrieval.py       # TF-IDF retriever
│   └── tools/
│       └── sqlite_tool.py     # DB utilities
├── data/
│   └── northwind.sqlite
├── docs/
│   ├── marketing_calendar.md
│   ├── kpi_definitions.md
│   ├── catalog.md
│   └── product_policy.md
├── sample_questions_hybrid_eval.jsonl
├── run_agent_hybrid.py
└── requirements.txt
```

