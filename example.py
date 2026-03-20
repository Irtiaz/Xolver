from xolver_tool import EpisodicMemory, Xolver

memory = EpisodicMemory(memory_file="example_memory.json")

# ── General ──────────────────────────────────────────────────────────────────

xolver = Xolver(model="qwen2.5-coder:7b", task_type="general", episodic_memory=memory)

answer = xolver.invoke("What were the main causes of World War I?")
print("=== General ===")
print(answer)

# ── Math ─────────────────────────────────────────────────────────────────────

xolver = Xolver(model="qwen2.5-coder:7b", task_type="math", episodic_memory=memory)

answer = xolver.invoke("Find the sum of all integers from 1 to 100.")
print("\n=== Math ===")
print(answer)

# ── Coding ───────────────────────────────────────────────────────────────────

xolver = Xolver(model="qwen2.5-coder:7b", task_type="coding", episodic_memory=memory)

answer = xolver.invoke(
    "Given two integers a and b, print their GCD.",
    test_cases=[
        {"input": "12 8",  "output": "4"},
        {"input": "100 75", "output": "25"},
        {"input": "7 13",  "output": "1"},
    ],
)
print("\n=== Coding ===")
print(answer)

# ── Coding (Python) ───────────────────────────────────────────────────────────

xolver = Xolver(model="qwen2.5-coder:7b", task_type="coding", language="python", episodic_memory=memory)

answer = xolver.invoke(
    "Given two integers a and b, print their GCD.",
    test_cases=[
        {"input": "12 8",   "output": "4"},
        {"input": "100 75", "output": "25"},
        {"input": "7 13",   "output": "1"},
    ],
)
print("\n=== Coding (Python) ===")
print(answer)
