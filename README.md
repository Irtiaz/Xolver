# Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team

### [Project Page](https://kagnlp.github.io/xolver.github.io/) | [Paper](https://arxiv.org/abs/2506.14234)

[Md Tanzib Hosain](https://scholar.google.com/citations?user=3YexY9gAAAAJ&hl=en),
[Salman Rahman](https://scholar.google.com/citations?user=vr7uTc8AAAAJ&hl=en&oi=ao),
[Md Kishor Morol](https://scholar.google.com/citations?user=pjn3jg4AAAAJ&hl=en),
[Md Rizwan Parvez](https://scholar.google.com/citations?user=KhC8rtcAAAAJ&hl=en)

---

## xolver_tool.py

`xolver_tool.py` wraps the Xolver multi-agent framework as a drop-in tool with the same logical interface as an LLM: give it a query, get back a text answer. It uses [Ollama](https://ollama.com) as the model backend so no API key is required.

### Installation

```bash
pip install openai rank-bm25 nltk
```

Ollama must be running locally with your chosen model pulled:

```bash
ollama serve
ollama pull llama3.2
```

For coding tasks, `g++` must be installed:

```bash
sudo apt install g++   # Ubuntu/Debian
brew install gcc       # macOS
```

---

### Quick start

```python
from xolver_tool import Xolver

xolver = Xolver(model="llama3.2")
print(xolver.invoke("Given two integers a and b, print their GCD."))
```

---

### Task types

`xolver_tool.py` supports three task types controlled by the `task_type` parameter.

#### Coding (default)
Agents produce C++ solutions. If `test_cases` are provided they are compiled with `g++` and executed for scoring; otherwise an LLM judge is used.

```python
xolver = Xolver(model="llama3.2", task_type="coding")

answer = xolver.invoke(
    "Given two integers a and b, print their GCD.",
    test_cases=[
        {"input": "12 8",   "output": "4"},
        {"input": "100 75", "output": "25"},
        {"input": "7 13",   "output": "1"},
    ],
)
print(answer)  # C++ source code
```

#### Math
Agents wrap their final answer in `\boxed{answer}`. An LLM judge scores correctness and the verifier extracts the boxed value.

```python
xolver = Xolver(model="llama3.2", task_type="math")

answer = xolver.invoke("Find the sum of all integers from 1 to 100.")
print(answer)  # "5050"
```

#### General
No output format constraints. Suitable for open-ended questions, explanations, and any non-math non-coding query. The best agent response is returned directly.

```python
xolver = Xolver(model="llama3.2", task_type="general")

answer = xolver.invoke("What were the main causes of World War I?")
print(answer)
```

---

### Persistent episodic memory

Pass a shared `EpisodicMemory` instance to accumulate experience across queries. Xolver retrieves similar past problems at inference time to guide agents.

```python
from xolver_tool import EpisodicMemory, Xolver

memory = EpisodicMemory(memory_file="my_memory.json")

xolver = Xolver(model="llama3.2", task_type="math", episodic_memory=memory)
xolver.invoke("What is 15% of 240?")
xolver.invoke("A train travels 300 km in 4 hours. What is its average speed?")
# my_memory.json is updated automatically after each call
```

---

### Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `model` | `"llama3.2"` | Ollama model name |
| `base_url` | `"http://localhost:11434/v1"` | Ollama API base URL |
| `api_key` | `"ollama"` | Placeholder — Ollama does not validate this |
| `task_type` | `"coding"` | `"coding"`, `"math"`, or `"general"` |
| `agents` | `2` | Number of parallel reasoning agents |
| `rounds` | `2` | Iterative refinement rounds per query |
| `retrieval_k` | `5` | Examples retrieved from episodic memory per query |
| `temperature` | `0.2` | Sampling temperature for all LLM calls |
| `episodic_memory` | `None` | Pre-built `EpisodicMemory` instance |
| `episodic_memory_file` | `None` | Path to JSON file for persistent episodic memory |
| `update_memory` | `True` | Whether to save each solved query to episodic memory |

---

### Full example

See `example.py` for a runnable script that exercises all three task types with a shared episodic memory.

---

## Citation

```bibtex
@article{hosain2025xolver,
    title={𝕏olver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team},
    author={Md Tanzib Hosain and Salman Rahman and Md Kishor Morol and Md Rizwan Parvez},
    journal={arXiv preprint},
    year={2025}
}
```
