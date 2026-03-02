# Xolver Project — Claude Code Memory

## What this project is
Artifact codebase for the paper "Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team". The original benchmark scripts (`gsm/`, `aime/`, `math/`, `lcb/`) are evaluation harnesses and should not be modified. All new work lives in `xolver_tool.py`.

## Key files
- `xolver_tool.py` — reusable Xolver module (the main deliverable of our session)
- `example.py` — demonstrates all three task types using `qwen2.5-coder:7b` via Ollama
- `gsm/`, `aime/`, `math/`, `lcb/` — original benchmark scripts, untouched

## xolver_tool.py design decisions

### LLM backend
- Uses the `openai` Python client pointed at Ollama (`base_url="http://localhost:11434/v1"`, `api_key="ollama"`)
- No real API key needed; Ollama does not validate it
- Model name is a constructor parameter defaulting to `"llama3.2"`

### Task types
Three supported values for `task_type`:
- `"coding"` (default) — agents produce C++ wrapped in ```cpp``` blocks; judged by compiling with `g++` and running against test cases; verifier extracts the code block
- `"math"` — agents produce answers wrapped in `\boxed{answer}`; judged by LLM; verifier extracts the boxed value
- `"general"` — no output format constraints; judged by LLM; verifier returns the best agent response directly (no extra LLM call)

### Code execution (coding task)
- Uses C++ (not Python like the original lcb benchmark)
- Compile step: `g++` with a temp `.cpp` file
- Run step: binary executed with test case input piped via stdin (not mocked like Python's `builtins.input`)
- Temp files cleaned up in both success and exception paths
- Requires `g++` installed on the system

### Memory
- `EpisodicMemory`: BM25-based retrieval over past problem-solution pairs; optionally persisted to a JSON file; shared across multiple `Xolver` instances by passing the same object
- `_SharedMemory`: internal per-query top-m store, not exported
- `update_memory=True` by default — each `invoke()` call adds to episodic memory and saves to disk if `memory_file` is set

### Interface
- `Xolver` class; `invoke(query, test_cases=None) -> str` is the public method
- Mirrors an LLM tool: one string in, one string out
- All Xolver config (agents, rounds, temperature, retrieval_k, etc.) lives in `__init__`

### What Xolver does NOT use
- LLM tool calling (no `tools=[...]` in API calls)
- "Tool use" is simulated: agents are prompted to embed code in their response, the framework extracts it with regex and executes it externally
