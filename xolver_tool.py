import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional

import nltk
import openai
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab/english/")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ========== PROMPT TEMPLATES ==========

_PLANNER_PROMPT = """
You are a planner to solve a {task_type} problem. Here is the problem for which you have to plan:
{problem}

First draft required strictly greater than {m} {task_type} specialized roles labeling "Specialized Roles" to solve the problem collaboratively with
reasoning behind your draft of each role. Format the roles clearly, for example:
Specialized Roles:
1. Role Name: Reasoning of what this agent should focus on.
2. Role Name: Reasoning...
...
m. Role Name: Reasoning...
m + 1. Role Name: Reasoning...
...

Then select exactly the highly {m} {task_type} influential roles labeling "Influential Roles" from the prior drafted "Specialized Roles" by re-checking the reasoning behind your selection and
assign the prior selected "Influential Roles" among exactly the {m} agents to solve the problem. Format the roles clearly, for example:
Influential Roles:
1. Role Name: Reasoning of what this agent should focus on.
2. Role Name: Reasoning...
...
m. Role Name: Reasoning...
"""

_DYNAMIC_AGENT_PROMPT_MATH = """
You are a {role}. Your task is to solve a {task_type} problem. Here is the problem that you have to
solve:
{problem}

You were also given a couple of similar problems to the problem above along
with their reasoning and solutions to aid you in solving the problem at hand. Here are the similar
problems you were given:
{external_retrieval}
{self_retrieval}

And here was your original response:
{prev_response}

Also here is the leading responses with execution results from the response store:
{response_store}

Think carefully about where you went wrong, relating with responses in the response store. Then, try to
fix the solution producing a thought later reply with a solution to be executed and judged again. You can
integrate a Python tool to execute the calculations while replying your solution if required.

Make sure to wrap your final answer in \\boxed{{answer}} block with the entire solution (in the final answer step).
"""

_DYNAMIC_AGENT_PROMPT_CODING = """
You are a {role}. Your task is to solve a {task_type} problem. Here is the problem that you have to
solve:
{problem}
{test_cases}

You were also given a couple of similar problems to the problem above along
with their reasoning and solutions to aid you in solving the problem at hand. Here are the similar
problems you were given:
{external_retrieval}
{self_retrieval}

And here was your original response:
{prev_response}

Also here is the leading responses with execution results from the response store:
{response_store}

Think carefully about where you went wrong, relating with responses in the response store. Then, try to
fix the solution producing a thought first then reply with a {language} solution to be executed and judged again.
Make sure to wrap your code in "```{code_block}```" block, and include exactly one
block of code with the entire solution (in the final code step).
"""

_JUDGE_PROMPT = """
You are a judge. Your task is to judge the candidate solution of a {task_type} problem. Here is the
problem for which the candidate solution you have to judge:
{problem}

And here is the candidate response which to judge:
{candidate_response}

Please produce a score labeling "Score" (if the response is correct, it should be 1 otherwise should be 0) with reasoning
behind your judgement of the candidate solution to the problem.
"""

_VERIFIER_PROMPT_MATH = """
You are an answer extractor. Your task is to extract answer from the response to a {task_type}
problem. Here is the response for which the answer you have to extract:
{response}

Please extract the answer only inside from the \\boxed{{answer}} block from the response.
"""

_VERIFIER_PROMPT_CODING = """
You are an answer extractor. Your task is to extract answer from the response to a {task_type}
problem. Here is the response for which the answer you have to extract:
{response}

Please extract the answer only inside from the "```{code_block}```" block from the response.
"""

_DYNAMIC_AGENT_PROMPT_GENERAL = """
You are a {role}. Your task is to answer the following query:
{problem}

You were also given a couple of similar queries along with their responses to aid you. Here they are:
{external_retrieval}
{self_retrieval}

And here was your original response:
{prev_response}

Also here are the leading responses from the response store:
{response_store}

Think carefully about where you went wrong, relating with responses in the response store. Then provide
your best revised answer.
"""

_SELF_RECALL_PROMPT = """
You are asked to recall from your own internal knowledge a relevant but distinct {task_type} problem labeling "Problem" and its solution labeling "Response",
different from the following problem:
{problem}

Please provide the recalled problem and its complete solution.
"""


# ========== MEMORY ==========

class EpisodicMemory:
    """BM25-based long-term memory that persists problem-solution pairs across calls."""

    def __init__(self, memory_file: Optional[str] = None):
        self.memory: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.memory_file = memory_file
        if memory_file:
            self._load_safe(memory_file)

    def _load_safe(self, filepath: str):
        if os.path.exists(filepath) and os.stat(filepath).st_size > 0:
            try:
                with open(filepath, "r") as f:
                    self.memory = json.load(f)
                self._rebuild_index()
            except Exception as e:
                print(f"Warning: Could not load episodic memory from {filepath}: {e}. Starting empty.")
                self.memory = []
                self.tokenized_corpus = []
                self.bm25 = None

    def _rebuild_index(self):
        self.tokenized_corpus = [
            word_tokenize(e["problem"].lower()) for e in self.memory if e.get("problem")
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(word_tokenize(query.lower()))
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"problem": self.memory[i].get("problem", ""), "solution": self.memory[i].get("solution", "")}
            for i in top_n if self.memory[i].get("solution")
        ]

    def update(self, problem: str, solution: str):
        if not problem or not solution:
            return
        self.memory.append({"problem": problem, "solution": solution})
        self._rebuild_index()

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.memory, f, indent=2)


class _SharedMemory:
    def __init__(self, capacity: int):
        self.memory: List[Dict] = []
        self.capacity = capacity

    def update(self, new_entries: List[Dict]):
        self.memory.extend(new_entries)
        self.memory = sorted(self.memory, key=lambda x: x["score"], reverse=True)[: self.capacity]

    def best(self) -> Optional[Dict]:
        return self.memory[0] if self.memory else None


# ========== XOLVER ==========

class Xolver:
    """
    Xolver wrapped as a reusable LLM-like tool.

    Usage:
        xolver = Xolver(model="llama3.2", task_type="math")
        answer = xolver.invoke("What is the sum of interior angles of a pentagon?")

    For coding tasks with test cases:
        answer = xolver.invoke(
            "Write a function that returns the nth Fibonacci number.",
            test_cases=[{"input": "5", "output": "5"}, {"input": "10", "output": "55"}],
        )
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        task_type: str = "coding",
        language: str = "cpp",
        agents: int = 2,
        rounds: int = 2,
        retrieval_k: int = 5,
        temperature: float = 0.2,
        episodic_memory: Optional[EpisodicMemory] = None,
        episodic_memory_file: Optional[str] = None,
        update_memory: bool = True,
    ):
        """
        Args:
            model:                Ollama model name (e.g. "llama3.2", "mistral", "qwen2.5").
            base_url:             Ollama API base URL. Default: "http://localhost:11434/v1".
            api_key:              API key placeholder (Ollama doesn't validate this). Default: "ollama".
            task_type:            "coding", "math", or "general". Controls prompts, judging, and answer extraction. Default: "coding".
            language:             For coding tasks only. "cpp" or "python". Controls the code block tag and execution engine. Default: "cpp".
            agents:               Number of dynamic reasoning agents (m in the paper). Default: 2.
            rounds:               Number of iterative refinement rounds (I in the paper). Default: 2.
            retrieval_k:          Number of episodic memory examples to retrieve per query. Default: 5.
            temperature:          Sampling temperature for all LLM calls. Default: 0.2.
            episodic_memory:      A pre-built EpisodicMemory instance to share across multiple Xolver
                                  instances. Takes priority over episodic_memory_file.
            episodic_memory_file: Path to a JSON file for persistent episodic memory. Created on first
                                  save if it doesn't exist.
            update_memory:        If True, adds the solved problem and final answer to episodic memory
                                  after each invoke() call. Default: True.
        """
        self.model = model
        self.task_type = task_type
        self.language = language
        self.agents = agents
        self.rounds = rounds
        self.retrieval_k = retrieval_k
        self.temperature = temperature
        self.update_memory = update_memory

        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

        if episodic_memory is not None:
            self.episodic_memory = episodic_memory
        else:
            self.episodic_memory = EpisodicMemory(memory_file=episodic_memory_file)

    # ---- internal helpers ----

    def _call(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def _extract_roles(self, planner_response: str) -> List[str]:
        roles = []
        section = re.search(
            r"Influential Roles:\s*(.+?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
            planner_response,
            flags=re.S | re.I,
        )
        if section:
            numbered = re.findall(r"\d+\.\s*([^:]+):", section.group(1).strip())
            seen: set = set()
            for r in numbered:
                r = r.strip()
                if r not in seen:
                    roles.append(r)
                    seen.add(r)
        if len(roles) < self.agents:
            roles.extend([f"Expert Agent {i + 1}" for i in range(self.agents - len(roles))])
        return roles[: self.agents]

    def _parse_score(self, score_str: str) -> int:
        match = re.search(r"Score:\s*([01])", score_str, flags=re.I)
        return int(match.group(1)) if match else 0

    def _self_recall(self, problem: str) -> str:
        prompt = _SELF_RECALL_PROMPT.format(task_type=self.task_type, problem=problem)
        return self._call([{"role": "user", "content": prompt}]).strip()

    def _extract_code(self, text: str) -> str:
        tag = re.escape(self.language)
        match = re.search(rf"```{tag}(.*?)```", text, flags=re.S | re.I)
        return match.group(1).strip() if match else ""

    def _run_code_and_score(self, code: str, test_cases: list) -> tuple:
        if self.language == "python":
            return self._run_python_and_score(code, test_cases)
        return self._run_cpp_and_score(code, test_cases)

    def _run_cpp_and_score(self, code: str, test_cases: list) -> tuple:
        passed = 0
        reasoning = []
        for idx, case in enumerate(test_cases):
            input_data = case.get("input", "")
            expected = case.get("output", "").strip()
            src_path = ""
            bin_path = ""
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
                    f.write(code)
                    src_path = f.name
                bin_path = src_path.replace(".cpp", "")
                compile_proc = subprocess.run(
                    ["g++", "-o", bin_path, src_path],
                    capture_output=True, text=True, timeout=10,
                )
                os.unlink(src_path)
                src_path = ""
                if compile_proc.returncode != 0:
                    reasoning.append(f"Test {idx + 1}: Compilation failed — {compile_proc.stderr.strip()}")
                    os.unlink(bin_path)
                    continue
                run_proc = subprocess.run(
                    [bin_path], input=input_data,
                    capture_output=True, text=True, timeout=5,
                )
                os.unlink(bin_path)
                bin_path = ""
                actual = run_proc.stdout.strip()
                if actual == expected:
                    passed += 1
                    reasoning.append(f"Test {idx + 1}: Passed")
                else:
                    reasoning.append(f"Test {idx + 1}: Failed — expected {expected!r}, got {actual!r}")
            except Exception as e:
                reasoning.append(f"Test {idx + 1}: Exception — {e}")
                for path in (src_path, bin_path):
                    if path and os.path.exists(path):
                        os.unlink(path)
        return passed, "\n".join(reasoning)

    def _run_python_and_score(self, code: str, test_cases: list) -> tuple:
        passed = 0
        reasoning = []
        for idx, case in enumerate(test_cases):
            input_data = case.get("input", "")
            expected = case.get("output", "").strip()
            src_path = ""
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                    f.write(code)
                    src_path = f.name
                run_proc = subprocess.run(
                    ["python3", src_path], input=input_data,
                    capture_output=True, text=True, timeout=10,
                )
                os.unlink(src_path)
                src_path = ""
                if run_proc.returncode != 0:
                    reasoning.append(f"Test {idx + 1}: Runtime error — {run_proc.stderr.strip()}")
                    continue
                actual = run_proc.stdout.strip()
                if actual == expected:
                    passed += 1
                    reasoning.append(f"Test {idx + 1}: Passed")
                else:
                    reasoning.append(f"Test {idx + 1}: Failed — expected {expected!r}, got {actual!r}")
            except Exception as e:
                reasoning.append(f"Test {idx + 1}: Exception — {e}")
                if src_path and os.path.exists(src_path):
                    os.unlink(src_path)
        return passed, "\n".join(reasoning)

    def _judge(self, problem: str, response: str, test_cases: Optional[list]) -> int:
        if self.task_type == "coding" and test_cases:
            code = self._extract_code(response)
            if code:
                score, _ = self._run_code_and_score(code, test_cases)
                return score
            return 0
        score_str = self._call([{"role": "user", "content":
            _JUDGE_PROMPT.format(task_type=self.task_type, problem=problem, candidate_response=response)
        }])
        return self._parse_score(score_str)

    def _verify(self, response: str) -> str:
        if self.task_type == "general":
            return response
        if self.task_type == "coding":
            prompt = _VERIFIER_PROMPT_CODING.format(task_type=self.task_type, response=response, code_block=self.language)
        else:
            prompt = _VERIFIER_PROMPT_MATH.format(task_type=self.task_type, response=response)
        return self._call([{"role": "user", "content": prompt}])

    def _agent_prompt(
        self,
        role: str,
        problem: str,
        ext_text: str,
        self_text: str,
        prev_response: str,
        response_store: str,
        test_cases_text: str,
    ) -> str:
        if self.task_type == "general":
            return _DYNAMIC_AGENT_PROMPT_GENERAL.format(
                role=role,
                task_type=self.task_type,
                problem=problem,
                external_retrieval=ext_text,
                self_retrieval=self_text,
                prev_response=prev_response or "None",
                response_store=response_store,
            )
        if self.task_type == "coding":
            lang_label = "Python" if self.language == "python" else "C++"
            return _DYNAMIC_AGENT_PROMPT_CODING.format(
                role=role,
                task_type=self.task_type,
                problem=problem,
                test_cases=test_cases_text,
                external_retrieval=ext_text,
                self_retrieval=self_text,
                prev_response=prev_response or "None",
                response_store=response_store,
                language=lang_label,
                code_block=self.language,
            )
        return _DYNAMIC_AGENT_PROMPT_MATH.format(
            role=role,
            task_type=self.task_type,
            problem=problem,
            external_retrieval=ext_text,
            self_retrieval=self_text,
            prev_response=prev_response or "None",
            response_store=response_store,
        )

    # ---- public interface ----

    def invoke(self, query: str, test_cases: Optional[list] = None) -> str:
        """
        Solve a problem using the Xolver multi-agent framework.

        Args:
            query:      The problem or question to solve.
            test_cases: For coding tasks only. A list of {"input": str, "output": str} dicts
                        used for execution-based scoring. When omitted, an LLM judge is used.

        Returns:
            The final extracted answer as a plain string.
        """
        shared_memory = _SharedMemory(capacity=self.agents)
        agent_responses = [""] * self.agents

        # Planner assigns roles
        planner_response = self._call([{"role": "user", "content":
            _PLANNER_PROMPT.format(task_type=self.task_type, problem=query, m=self.agents)
        }])
        roles = self._extract_roles(planner_response)

        # Episodic retrieval happens once before all rounds
        retrieved = self.episodic_memory.retrieve(query, k=self.retrieval_k)

        test_cases_text = ""
        if self.task_type == "coding" and test_cases:
            test_cases_text = f"\nHere are the test cases:\n{json.dumps(test_cases, indent=2)}"

        # Iterative refinement rounds
        for _ in range(self.rounds):
            response_store = "\n".join(
                f"Agent: {e['agent']}\nResponse: {e['response']}\nScore: {e['score']}"
                for e in shared_memory.memory
            ) or "None"

            for i, role in enumerate(roles):
                if retrieved:
                    ext_text = "\n\n".join(
                        f"Problem:\n{e['problem']}\n\nResponse:\n{e['solution']}"
                        for e in retrieved
                    )
                    self_text = "None"
                else:
                    ext_text = "None"
                    self_text = self._self_recall(query)

                response = self._call([{"role": "user", "content":
                    self._agent_prompt(
                        role=role,
                        problem=query,
                        ext_text=ext_text,
                        self_text=self_text,
                        prev_response=agent_responses[i],
                        response_store=response_store,
                        test_cases_text=test_cases_text,
                    )
                }])
                score = self._judge(query, response, test_cases)
                agent_responses[i] = response
                shared_memory.update([{"agent": role, "response": response, "score": score}])

        # Verifier extracts the final answer from the best response
        best = shared_memory.best() or {"response": agent_responses[0] if agent_responses else ""}
        final_answer = self._verify(best["response"])

        if self.update_memory:
            self.episodic_memory.update(query, final_answer)
            if self.episodic_memory.memory_file:
                self.episodic_memory.save(self.episodic_memory.memory_file)

        return final_answer
