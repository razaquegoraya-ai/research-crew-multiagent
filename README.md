# research-crew

A **multi-agent research & report generation system**. Give it a topic or a
company; a small crew of specialized [CrewAI](https://www.crewai.com/) agents
collaborates to research it on the live web and produce a finished, **sourced
markdown report** — with every agent hand-off recorded to a structured trace so
you can *see* the orchestration happening.

> Built as a portfolio piece: the code is modular and typed, and the
> orchestration trace is treated as a first-class, visible feature.

---

## What it does

1. A **Research Lead** breaks your topic into 3–5 focused sub-questions.
2. A **Web Researcher** searches the live web (via [Tavily](https://tavily.com))
   for each sub-question and returns findings **with source URLs**.
3. An **Analyst** synthesizes the findings into structured insight and flags
   conflicting information.
4. A **Report Writer** drafts a clean markdown report.
5. A lightweight **review step** (the Analyst as fact-checker) flags unsupported
   claims, and the Writer produces the **final, corrected report**.

Every run writes two files to `outputs/`:

- `report_<timestamp>.md` — the finished report.
- `trace_<timestamp>.json` — the full, ordered hand-off trace (which agent ran,
  its input, its output, and timing). The same trace streams to your console
  live.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
        topic, depth     │                 ResearchCrew                  │
   user ───────────────▶ │  (assembles LLM, tools, agents, tasks, trace) │
                         └───────────────────────┬──────────────────────┘
                                                 │ sequential process
                                                 ▼
   ┌────────────┐   ┌───────────────┐   ┌───────────┐   ┌──────────────┐
   │ Research   │──▶│ Web Researcher │──▶│  Analyst  │──▶│ Report Writer │
   │ Lead       │   │ (Tavily tool)  │   │           │   │  (draft)      │
   │ sub-Qs     │   │ sourced finds  │   │ synthesis │   └──────┬───────┘
   └────────────┘   └───────┬────────┘   └─────┬─────┘          │
                            │                  │                ▼
                            │                  │         ┌──────────────┐
                            │                  └────────▶│  Analyst     │
                            │                    review  │ (fact-check) │
                            │                            └──────┬───────┘
                            ▼                                   ▼
                   ┌──────────────────┐              ┌──────────────────┐
                   │  SourceCollector │              │  Report Writer   │
                   │ (unique sources) │              │  (finalize)      │
                   └──────────────────┘              └────────┬─────────┘
                                                              │
   ┌───────────────────────────────────────────────┐         ▼
   │  Tracer  (task_callback + step_callback)        │   outputs/report_*.md
   │  console log  +  outputs/trace_<ts>.json  ⭐     │   outputs/trace_*.json
   └───────────────────────────────────────────────┘

   Access layers:  main.py (CLI)   ·   FastAPI  POST /research  ·  GET /health
```

### Project structure

```
research-crew/
├── main.py                       # CLI entry point
├── requirements.txt              # pinned dependencies
├── .env.example                  # required config keys
├── README.md
├── outputs/                      # generated reports + traces (samples committed)
│   ├── report_20260917_120000_sample.md
│   └── trace_20260917_120000_sample.json
├── research_crew/
│   ├── config.py                 # env-driven settings (pydantic-settings)
│   ├── logging_config.py         # shared console logging setup
│   ├── crew.py                   # ResearchCrew: assemble + run + persist
│   ├── models/schemas.py         # Pydantic models (request, source, trace, …)
│   ├── agents/crew_agents.py     # the 4 agent definitions
│   ├── tasks/crew_tasks.py       # the task graph incl. the review step
│   ├── tools/search_tool.py      # Tavily tool + SourceCollector
│   ├── tracing/tracer.py         # ⭐ first-class hand-off tracer
│   ├── api/app.py                # FastAPI application (+ serves the UI)
│   └── web/index.html           # single-page dashboard UI
└── tests/test_smoke.py           # offline tests (no API keys needed)
```

---

## Setup

Requires **Python 3.11+**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
#   then edit .env and set OPENAI_API_KEY and TAVILY_API_KEY
```

Required keys (see [`.env.example`](.env.example)):

| Variable          | Purpose                                  |
|-------------------|------------------------------------------|
| `OPENAI_API_KEY`  | LLM for every agent (GPT-4o by default). |
| `TAVILY_API_KEY`  | Live web search. Free tier at tavily.com |

Keys are read from the environment only — nothing is hardcoded.

---

## Sample run

### CLI

```bash
python main.py -t "The competitive landscape for AI code assistants in 2026" -d deep
```

You'll see the hand-off trace stream live, e.g.:

```
12:00:00 | INFO    | research_crew.crew  | Starting research run '20260917_120000_...' (depth=deep)
12:00:11 | INFO    | research_crew.trace | ──▶ [00] HAND-OFF → Research Lead | plan_sub_questions | 11.20s
12:00:18 | INFO    | research_crew.trace |   ↳ [01] tool_call: web_search (6.70s)
...
12:01:52 | INFO    | research_crew.trace | ──▶ [10] HAND-OFF → Report Writer | finalize_report | 7.40s
12:01:52 | INFO    | research_crew.crew  | Run '...' complete: 14 sources, 11 trace events, 112.4s

======================================================================
RESEARCH COMPLETE
======================================================================
Topic          : The competitive landscape for AI code assistants in 2026
Depth          : deep
Agents used    : Research Lead, Web Researcher, Analyst, Report Writer
Sources        : 14
Report file    : outputs/report_20260917_120000_....md
Trace file     : outputs/trace_20260917_120000_....json
======================================================================
```

### API

```bash
uvicorn research_crew.api.app:app --reload
```

Then:

```bash
curl -s http://localhost:8000/health | jq

curl -s -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Anthropic'\''s position in the AI market", "depth": "quick"}' | jq
```

`POST /research` returns:

```json
{
  "report_markdown": "# ...full markdown report...",
  "metadata": {
    "topic": "...",
    "depth": "quick",
    "model": "gpt-4o",
    "agents_used": ["Research Lead", "Web Researcher", "Analyst", "Report Writer"],
    "sub_questions_count": 3,
    "sources_count": 9,
    "elapsed_seconds": 64.2,
    "report_file": "outputs/report_....md",
    "trace_file": "outputs/trace_....json",
    "trace_event_count": 8
  },
  "sources": [ { "url": "https://...", "title": "...", "snippet": "..." } ]
}
```

Interactive API docs are at `http://localhost:8000/docs`.

### Web dashboard

Once the API is running, open **`http://localhost:8000/`** for the built-in
single-page dashboard (no build step, served by FastAPI):

- Enter a topic, pick **quick**/**deep**, and hit **Run research** to call the
  live API.
- Or click **Load sample run** to explore a full result instantly — no API
  keys required — using the committed sample data via `GET /api/sample`.
- Results render as three tabs: the **Report** (rendered markdown), the
  **Agent trace** (a color-coded, expandable hand-off timeline with per-event
  timing — the visual proof of orchestration), and **Sources**.

The UI lives in [`research_crew/web/index.html`](research_crew/web/index.html).

---

## Sample output

A committed, illustrative example lives in
[`outputs/report_20260917_120000_sample.md`](outputs/report_20260917_120000_sample.md)
with its matching trace at
[`outputs/trace_20260917_120000_sample.json`](outputs/trace_20260917_120000_sample.json).

> These two files were hand-authored to show the output *shape and quality*;
> they were not produced by a live run. Add your keys and run the command above
> to generate real ones.

---

## Design notes

- **The trace is a product feature.** `Tracer` hooks into CrewAI's
  `task_callback` and `step_callback` to record every hand-off and tool call
  with timing, streaming to the console and persisting to JSON.
- **Sources are counted, not guessed.** The search tool writes every retrieved
  result into a thread-safe `SourceCollector`, so `sources_count` reflects real
  retrievals rather than parsed LLM text.
- **Sequential process.** The crew runs its six tasks in a fixed order so the
  trace is deterministic and easy to read — ideal for a demo.
- **Graceful failure.** Missing keys fail fast with a clear message (HTTP 503
  via the API); Tavily rate limits are retried with backoff; empty results
  return a model-readable note instead of crashing.

---

## Testing

Offline tests (no API keys or network required):

```bash
pytest -q
```

They cover model validation, config secret-checking, the tracer's
recording/serialization, and the search tool's empty-result and dedup handling
(the search-tool tests skip automatically if CrewAI isn't installed).

---

## Tuning

| Setting              | Env var              | Default | Effect                              |
|----------------------|----------------------|---------|-------------------------------------|
| Model                | `OPENAI_MODEL`       | `gpt-4o`| LLM all agents run on               |
| Temperature          | `LLM_TEMPERATURE`    | `0.2`   | Lower = more deterministic          |
| Results per question | `SEARCH_MAX_RESULTS` | `5`     | Tavily results fetched per query    |
| Output directory     | `OUTPUT_DIR`         | `outputs` | Where reports/traces are written  |
| Log level            | `LOG_LEVEL`          | `INFO`  | Console verbosity                   |

Depth also tunes behavior: `quick` → 3 sub-questions, `deep` → 5.

---

## License

Released under the [MIT License](LICENSE) © 2026 Abdul Razaque Goraya.
