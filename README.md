# Qwen Extraction Agent

An autonomous AI data-extraction agent foundation. This repository is the
**Phase 1 base project only** — a small, honest, runnable slice of a much
larger planned system, built to be incrementally expanded.

> **Status: incremental project.** Only HTTP/file-based extraction (Phase 1)
> is implemented. Browser automation, email, SPARQL/API querying, and
> production infrastructure (database, monitoring, scheduling) are **not**
> implemented — see [Roadmap](#roadmap).

## 1. What this project is

A single AI agent (Qwen3-8B) that extracts data from heterogeneous sources
by choosing deterministic tools, observing their results, and deciding
what to do next — instead of a human hand-writing a scraping "recipe" for
every source.

## 2. Why it exists

The predecessor to this project is a v2 pipeline built around per-source,
hand-written recipes (Playwright/HTTP/API scripts). That approach doesn't
scale cleanly to 150+ heterogeneous sources with different access
patterns (direct download, dynamic filters, email delivery, query APIs,
traditional scraping). This repository converts that recipe-based
pipeline into an **agent architecture**, while keeping the parts of the v2
pipeline that were already correct: per-dataset isolation, raw-artifact
preservation, deterministic parsing, validation, and auditability.

## 3. Architecture

```
SOURCE CONFIG
      |
      v
   AI AGENT (Qwen3-8B) -- chooses a tool + arguments as JSON
      |
      v
CONTROLLED TOOL EXECUTION (execute_action)
      |
      v
   TOOL RESULT (observation)
      |
      v
   AI AGENT -- chooses next tool, or "finish"
      |
     ...
      v
  extract -> validate -> save raw + processed data
```

Strict separation of concerns:

```
MODEL -> AGENT -> TOOL REGISTRY -> TOOLS -> EXTRACTION/PARSING -> VALIDATION -> STORAGE
```

* The model never knows how `requests.get()`, `pandas`, or `pdfplumber`
  work internally.
* The agent loop (`agent/loop.py`) contains no dataset-specific logic.
* Tools (`tools/`) contain no LLM reasoning.
* Source configs (`config/sources/*.yaml`) describe a source declaratively
  — they never contain executable Python.

## 4. Agent philosophy

> AI decides what to do. Tools deterministically do it. AI observes what
> happened. AI decides what to do next.

The model **never** executes arbitrary Python, shell commands, or SQL. It
returns strict JSON (`{"action": ..., "arguments": ...}`); the tool
registry validates the action and arguments against a JSON Schema before
anything runs. Malformed model output is never silently executed — it
becomes a structured `PARSE_ERROR` observation the model sees on its next
turn.

This is a **single agent**. There are no planner/researcher/browser/
validator sub-agents. Tools provide capabilities; Qwen is the only
decision maker.

## 5. Model: Qwen3-8B

* Hugging Face model: `Qwen/Qwen3-8B`
* Runs in 4-bit quantization (bitsandbytes) — proven on an NVIDIA Tesla T4
  (14.56 GB VRAM) in Google Colab, using roughly 5.7 GB VRAM.
* `enable_thinking=False` is set explicitly via the Qwen3 chat template
  for tool-selection calls, so the model returns only JSON with no
  `<think>` preamble. This is **not** approximated by truncating with
  `max_new_tokens`.
* The generation config is per-call-mode (`agent/model.py:
  GENERATION_CONFIGS`), so future phases (extraction reasoning, schema
  mapping, recovery) can use different settings without changing the
  agent loop.
* Nothing in this repository hard-codes Google Colab. Colab is the current
  *development* environment; `agent/model.py` works anywhere with a
  CUDA GPU (and PyTorch/transformers/bitsandbytes installed).

## 6. Controlled tools

Every capability the model can invoke is registered as a `ToolSpec`
(`tools/registry.py`): name, description, category, JSON-Schema argument
spec, the actual function, and an `enabled` flag. `execute_action()` is
the **only** path from a model decision to real execution:

1. Special-cases `finish` (no function execution).
2. Looks up the tool (`ToolNotFoundError` if unknown).
3. Confirms it's enabled (`ToolDisabledError` if not — this is how
   not-yet-implemented Phase 2-4 tools are represented).
4. Validates arguments against the tool's JSON Schema
   (`InvalidArgumentsError` if invalid).
5. Executes the tool, catching runtime failures into a structured
   `{"success": false, "error_type": ..., "message": ..., "recoverable": ...}`
   result instead of crashing the agent.

The model can never reach `eval`, `exec`, a shell, an unrestricted
subprocess, or arbitrary SQL — those are not tools, and there is no
generic "run code" tool in the registry.

## 7. Five source classes (proving set)

| Class | Description | Representative source | Status |
|---|---|---|---|
| 1a | Direct-download datasets | Train Name Index | Phase 1 target |
| 1b | Dynamic-filter downloads | Daily UPI | Phase 2 (browser) |
| 1c | Email-triggered downloads | FSSAI | Phase 4 (email) |
| 1d | Query-based extraction | Wikidata | Phase 3 (SPARQL) |
| 2  | Traditional web scraping | Elections | Phase 2 (browser) |

This repository proves class **1a only**, via `http_download`.

## 8. Development phases

* **Phase 1 (this repository): HTTP/file-based extraction foundation.**
  `http_download`, `inspect_file`, `read_excel`, `read_csv`, `read_pdf`,
  `extract_pdf_table`, plus deterministic validation helpers.
* **Phase 2: Browser/Playwright.** `browser_open`, `browser_inspect`,
  `browser_click`, `browser_fill`, `browser_select`, `browser_wait`,
  `browser_download`, `browser_back`.
* **Phase 3: Query/API.** `sparql_query`, `api_get`.
* **Phase 4: Email.** `email_search`, `email_read`, `email_get_attachment`,
  `email_download_link`.
* **Phase 5: Production infrastructure.** Hash gate, raw/processed
  storage conventions (paths already scaffolded), database, monitoring,
  status, scheduling, retry/recovery, recipe generation/update.

Phase 2-4 tool names are registered in `tools/definitions.py` with
`function=None` and `enabled=False` purely so the model's tool listing is
honest about what's planned. Their bodies live in `tools/future_tools.py`
and raise `NotImplementedError` — they are never wired to run.

## 9. Repository structure

```
qwen-extraction-agent/
|-- agent/
|   |-- state.py       # AgentState, AgentStatus, ToolCallRecord
|   |-- model.py        # ModelClient protocol + QwenModelClient
|   |-- prompts.py       # system/user prompt building + strict JSON parsing
|   `-- loop.py          # run_agent(): the control loop
|-- tools/
|   |-- registry.py      # ToolSpec, ToolRegistry, execute_action
|   |-- http_tools.py    # http_download (Phase 1)
|   |-- file_tools.py    # inspect_file, read_csv, read_excel, read_pdf, extract_pdf_table
|   |-- validation_tools.py # validate_* helpers
|   |-- future_tools.py  # Phase 2-4 placeholders (NotImplementedError)
|   `-- definitions.py   # build_registry(): assembles everything above
|-- core/
|   |-- config_loader.py # load_source_config()
|   |-- logging_utils.py # get_logger(), log_tool_call()
|   `-- exceptions.py    # ProjectError and subclasses
|-- storage/
|   |-- paths.py         # raw_path()/processed_path() helpers
|   |-- raw/              # downloaded artifacts (gitignored, .gitkeep only)
|   |-- processed/        # processed output (gitignored, .gitkeep only)
|   `-- logs/              # tool_calls.jsonl audit log (gitignored)
|-- config/sources/
|   `-- test_direct_download.yaml  # the Phase 1 proving source config
|-- tests/
|   |-- unit/             # no GPU/model required
|   `-- integration/      # full agent loop with a scripted model double
|-- notebooks/
|   `-- 01_qwen_agent_test.ipynb  # Colab: real Qwen3-8B end-to-end
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- pytest.ini
`-- conftest.py
```

## 10. Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HUGGINGFACE_TOKEN if your account needs it
```

## 11. Running locally vs. Colab

* **Locally, without a GPU:** you can run everything except the real
  Qwen model — all unit tests, the tool registry, `execute_action`, file
  tools, and the agent loop driven by a scripted `ModelClient` double
  (see `tests/integration/test_qwen_agent_integration.py`).
* **On Colab (or any CUDA GPU host):** open
  `notebooks/01_qwen_agent_test.ipynb` to run the real Qwen3-8B model
  end-to-end against the proving source.

## 12. First proving test

The first milestone this repository proves:

```
config/sources/test_direct_download.yaml
        |
        v
      Qwen3-8B  ->  {"action": "http_download", "arguments": {...}}
        |
        v
   execute_action()  ->  http_download(https://httpbin.org/bytes/100)
        |
        v
   structured tool result  ->  {"success": true, "bytes": 100, ...}
```

Run it with the real model via the Colab notebook, or exercise the exact
same code path without a GPU using the scripted-model integration test:

```bash
pytest tests/integration/test_qwen_agent_integration.py -m network
```

## 13. Testing

```bash
# All tests that don't require network or a GPU/model
pytest -m "not network and not model"

# Everything, including the live httpbin.org download test
pytest -m "not model"

# Only unit tests
pytest tests/unit

# Only the integration loop test (network required, no GPU required)
pytest tests/integration
```

Markers (defined in `pytest.ini`):

* `network` — requires outbound access to httpbin.org.
* `model` — requires loading the real Qwen3-8B model on a GPU. No test in
  this repository currently carries this marker; it's reserved for the
  Colab notebook's manual workflow rather than automated CI.

## 14. Roadmap

1. Multi-step agent loop already supports it structurally; next is
   wiring the **Train Name Index** source through the full
   download -> inspect -> parse -> validate -> finish sequence
   end-to-end with the real model.
2. Phase 2: implement `browser_open`/`browser_download`/etc. with
   Playwright for **Daily UPI** (dynamic-filter downloads).
3. Phase 3: implement `sparql_query` for **Wikidata**.
4. Phase 4: implement email tools for **FSSAI**.
5. Phase 5: hash gate, TimescaleDB/PostgreSQL storage, monitoring,
   scheduling, and recipe generation/update.

## 15. Security principles

* The model **never** has access to `eval`, `exec`, a shell, unrestricted
  subprocess execution, arbitrary SQL, or unrestricted filesystem
  traversal. It can only select from explicitly registered tools.
* Every tool declares its own JSON-Schema argument contract; arguments
  are validated before the tool function runs.
* Secrets (Hugging Face tokens, email credentials, database URLs) are
  never hard-coded — see `.env.example` — and `.env` is gitignored.
* Raw artifacts and processed data are kept in separate storage roots
  (`storage/raw/` vs `storage/processed/`); nothing in this repository
  overwrites a raw artifact with processed output.
* Every executed tool call is logged as a structured, auditable JSON
  line (`storage/logs/tool_calls.jsonl`) with timestamp, source, step,
  tool name, arguments, result status, error, and duration.
