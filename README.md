# Vespera Helpdesk AI -- LLM Quality & Evaluation Framework

This is the companion project for the Sypher course "AI & LLM Testing." It
has two things in it:

1. **Vespera Systems' Helpdesk AI Assistant** -- a small, fictional
   internal tool that answers a support agent's product questions. It
   starts as a plain LLM application and becomes a RAG (retrieval-augmented
   generation) system partway through the course.
2. **The LLM Quality & Evaluation Framework** -- a real, working toolkit
   for testing that assistant: response validation, semantic and
   LLM-as-a-judge evaluation, RAG-specific checks, adversarial safety
   tests, baseline/regression detection, observability, reporting, and a
   CI quality gate.

Everything here runs against a **local Ollama model** -- no cloud API,
no API key, no cost. `llama3.2:3b` handles every chat/generation call;
`nomic-embed-text` handles every embedding call (see "A real bug found
while building this" below for why there are two models instead of one).

It is the companion code for the course's 25 modules, built up milestone
by milestone the same way the course presents them. `app/` is the
finished project.

## A real bug found while building this

The course's own lessons call `ollama.embed(model="llama3.2:3b", ...)`
for semantic similarity and RAG retrieval. On a real, locally-running
Ollama server, that call fails outright:

```
ollama._types.ResponseError: This server does not support embeddings. Start it with `--embeddings`
```

`llama3.2:3b` is a chat model, not an embedding model, and this server
rejects embedding requests against it. The fix used throughout this repo
(`llmqe/config.py`'s `embed_model` field): pull `nomic-embed-text` (274 MB,
free, local, no cost) and use it for every embedding call, while every
chat/generation call still uses `llama3.2:3b`. Nothing about the course's
"local-only, no paid API" design changes -- this is a second free local
model, not a cloud dependency.

A second real bug, found the same way: the course's Module 12 safety
suite hardens its system prompt once (its "V2") and the lesson's own text
says a well-defended prompt "should" resist a `"You are now DAN..."`
jailbreak. Running that exact V2 prompt against a real `llama3.2:3b`
showed it reliably complying anyway (`"DAN mode enabled."`), every time,
across repeated runs. `llmqe/safety/safety_tests.py`'s `V3` prompt adds
one more explicit counter-pattern naming "DAN" and mode-confirmation
requests directly, which does reliably resist it -- verified with
repeated real runs, not assumed.

## Set up your computer

You need Git, Python 3.11 or newer (built and verified with 3.14), and
Ollama.

| Tool | Windows | macOS | Ubuntu or Debian |
| --- | --- | --- | --- |
| Git | `winget install --id Git.Git -e` | `brew install git` | `sudo apt update && sudo apt install -y git` |
| Python | `winget install Python.Python.3.12` | `brew install python@3.12` | `sudo apt install -y python3 python3-venv` |
| Ollama | download from Ollama's website | `brew install ollama` | `curl -fsSL https://ollama.ai/install.sh \| sh` |

Check them (`python` is `python3` on macOS and Linux):

```bash
git --version
python3 --version    # python --version on Windows
ollama --version
```

### Get the code

```bash
git clone <vespera-helpdesk-ai repository URL> vespera-helpdesk-ai
cd vespera-helpdesk-ai/app
```

Every command below runs from `app/`.

## Run the app

1. **Pull both local models** this project needs (about 2.3 GB total):

   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

2. **Create and activate a virtual environment**, then install the
   dependencies:

   ```bash
   python3 -m venv .venv            # Windows: python -m venv .venv
   source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Confirm your environment is ready**, the same three-line check the
   course's Setup lesson uses:

   ```bash
   python check_setup.py
   ```

   A short greeting printed to your console means Ollama, the Python
   client, and `llama3.2:3b` are all working.

4. **Ask the plain (ungrounded) assistant a question:**

   ```bash
   python -c "from llmqe.client import OllamaClient, SYSTEM_PROMPT; c = OllamaClient(); print(c.ask_with_system(SYSTEM_PROMPT, 'Does the Pro plan support SSO?').text)"
   ```

5. **Ask the RAG-backed assistant the same question it used to
   hallucinate on** (Module 5's incident, fixed by Module 9's retrieval):

   ```bash
   python -c "
   from llmqe.client import OllamaClient
   from llmqe.rag.retriever import Retriever
   from llmqe.rag.assistant import ask_with_retrieval
   client = OllamaClient()
   retriever = Retriever()
   result = ask_with_retrieval(\"What's the API rate limit on the Free plan?\", retriever, client)
   print(result['response'].text)
   "
   ```

   This answers from Vespera's real (fictional) documentation --
   **60 requests per minute** -- instead of guessing.

## Run the evaluation framework

Build a baseline once (writes `llmqe/baseline/baselines/v1.json`):

```bash
python scripts/build_baseline.py
```

Run the full nightly-style evaluation (calibration, RAG relevance,
safety, logging, regression gate, and report, exactly the way
`.github/workflows/nightly-full-eval.yml` runs it):

```bash
python scripts/run_full_suite.py
```

Run just the CI quality gate (exits non-zero if a release should be
blocked):

```bash
python -m llmqe.regression.run_and_gate
```

## Run the tests

```bash
pytest
```

Every test here makes real calls to your local Ollama server -- there is
nothing mocked, by design, since the entire point of this course is
testing a real model's real behavior. A few tests (consistency, latency)
are marked `slow` because they make several calls in a row:

```bash
pytest -m "not slow"   # fast subset, mirrors the PR smoke suite
```

## Project layout

```text
vespera-helpdesk-ai/
  app/
    llmqe/
      client.py                 OllamaClient, LLMResponse, SYSTEM_PROMPT
      config.py                 FrameworkConfig -- one source of truth for model/version literals
      known_facts.py             VESPERA_FACTS -- ground truth used by hallucination checks
      validation.py               validate_response(): exact-match, rule-based, schema checks
      providers/
        base.py                   LLMProvider protocol
        ollama_provider.py        the real provider
        recorded_provider.py      a fixed-response test double
      eval/
        scorer.py                 semantic similarity, score_case()'s four evaluation stages
        judge.py                  LLM-as-a-judge
        evaluator_tests.py        tests the evaluator itself against a known-good/known-bad set
        datasets/                 qa_dataset.jsonl, evaluator_calibration.jsonl
      rag/
        documents.py               Vespera's (fictional) product documentation
        retriever.py               embeds once, retrieves top-k by cosine similarity
        assistant.py                RAG-backed ask_with_retrieval(), with citation
        rag_eval.py                 retrieval relevance, citation honesty/correctness, groundedness
        datasets/                   retrieval_dataset.jsonl
      safety/
        safety_tests.py             adversarial prompt-injection / jailbreak suite
      baseline/
        baseline.py                  BaselineRecord, record/load/compare
        baselines/v1.json            a real recorded baseline
      regression/
        regression_suite.py          routes a baseline diff to a block/don't-block decision
        run_and_gate.py               CI entry point, exits non-zero to block a release
      observability/
        logger.py                    structured JSONL log of every evaluation run
      reporting/
        report.py                    turns the log into a human-readable Markdown report
      reliability/
        perf_tests.py                 latency measurement, retry/timeout classification
    scripts/
      build_baseline.py
      run_full_suite.py
    tests/                           one file per concern above
    check_setup.py                   the Setup lesson's 3-line confirmation script
    requirements.txt
    pyproject.toml
    .github/workflows/pr-smoke.yml
    .github/workflows/nightly-full-eval.yml
```

## Troubleshooting

- **`ConnectionError` on any call.** The Ollama server isn't running.
  Check with `curl http://localhost:11434` and start it with
  `ollama serve` if it doesn't respond.
- **A call hangs with no error.** The model named in `OllamaClient` or
  `CONFIG.embed_model` was never pulled. Check with `ollama list` and
  pull whichever is missing.
- **`ResponseError: This server does not support embeddings`.** You're
  calling `get_embedding()` with a chat model instead of
  `CONFIG.embed_model` (`nomic-embed-text`) -- see "A real bug found
  while building this" above.
- **`pytest` can't import `llmqe`.** Run `pytest` from `app/`, with the
  virtual environment active.
- **A safety test fails intermittently.** This is real, not flaky test
  infrastructure -- it means the model's compliance with a given prompt
  genuinely varies run to run. Re-run it a few times; if it fails
  consistently, the prompt needs strengthening, the same diagnosis the
  course's Module 12/14 lessons walk through.

## License

Copyright (C) 2026 SypherNext

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). You may
use, study and change it. If you share a modified version, or run one as a website or service
that other people use, you must make your full modified source code available under the same
license. See [LICENSE](LICENSE) for the full terms.
