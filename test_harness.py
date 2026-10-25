"""
Auto-generates a runnable test harness around the candidate's raw
Solution-class code, LeetCode-style: the candidate pastes only their
class/function, and this builds the driver code by parsing the worked
example(s) already stated in the question — so grading can report
actual pass/fail per test case instead of just "did it compile".

Only works when the question text contains a worked example
(Input:/Output:). Codeforces-sourced questions (link-only metadata, no
fetched problem statement) and the hardcoded fallback bank don't have
one, so this cleanly raises and the caller (interviewer.py) falls back
to the plain "paste a full runnable program yourself" flow.
"""

import re
from langchain_groq import ChatGroq

_llm = None

_PROMPT = """You are building an automated test harness for a coding interview question.

Question (should include at least one worked example with Input/Output):
{question}

Candidate's submitted code — do NOT change their algorithm logic at all:
{candidate_code}

Task:
1. Identify every worked example test case stated in the question.
2. Write a COMPLETE, runnable {language} program that:
   - Includes the candidate's code EXACTLY as given, unchanged.
   - Adds whatever driver code is needed to construct each example's input,
     call the candidate's method/function, and print ONLY the result of
     each test case, one per line, each prefixed with "RESULT: ".
   - Prints nothing else to stdout.
3. List the expected output for each test case, in the same order as the
   RESULT: lines the program will print, one per line.

If the question has no worked example with a clear input and output,
respond with exactly: NO_EXAMPLE_FOUND

Otherwise respond in exactly this format, no other text:
---CODE---
<the complete program>
---EXPECTED---
<expected output for test case 1>
<expected output for test case 2>
"""


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
    return _llm


def _strip_result_prefix(line: str) -> str:
    """The prompt tells the LLM not to include 'RESULT:' in expected outputs,
    but it doesn't always follow that — strip it defensively rather than
    trusting the instruction, same lesson as the sandbox status-cap fix."""
    line = line.strip()
    if line.upper().startswith("RESULT:"):
        return line[len("RESULT:"):].strip()
    return line


def generate_harness(question: str, candidate_code: str, language: str) -> tuple[str, list[str]]:
    prompt = _PROMPT.format(question=question, candidate_code=candidate_code, language=language)
    response = _get_llm().invoke(prompt).content.strip()

    if "NO_EXAMPLE_FOUND" in response:
        raise ValueError("No worked example in the question to build a test harness from")
    if "---CODE---" not in response or "---EXPECTED---" not in response:
        raise ValueError("Harness generation returned an unexpected format")

    code_part = response.split("---CODE---", 1)[1].split("---EXPECTED---", 1)[0].strip()
    expected_part = response.split("---EXPECTED---", 1)[1].strip()
    expected_outputs = [
        _strip_result_prefix(line) for line in expected_part.splitlines() if line.strip()
    ]

    if not code_part or not expected_outputs:
        raise ValueError("Harness generation produced empty code or expected outputs")

    return code_part, expected_outputs


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def run_test_cases(question: str, candidate_code: str, language: str) -> dict:
    """Returns {tests_total, tests_passed, status, stderr, compile_output, details}.
    Raises if harness generation itself fails (no example, bad LLM output) —
    the caller decides how to fall back."""
    from tools.sandbox import run_code

    harness_code, expected_outputs = generate_harness(question, candidate_code, language)
    result = run_code(harness_code, language)

    if result["status"] != "Accepted":
        return {
            "tests_total": len(expected_outputs),
            "tests_passed": 0,
            "status": result["status"],
            "stderr": result["stderr"],
            "compile_output": result["compile_output"],
            "details": [],
        }

    actual_lines = [
        line.replace("RESULT:", "", 1).strip()
        for line in result["stdout"].splitlines()
        if line.strip().startswith("RESULT:")
    ]

    details = []
    passed = 0
    for i, expected in enumerate(expected_outputs):
        actual = actual_lines[i] if i < len(actual_lines) else None
        is_pass = actual is not None and _normalize(actual) == _normalize(expected)
        if is_pass:
            passed += 1
        details.append({"expected": expected, "actual": actual, "passed": is_pass})

    return {
        "tests_total": len(expected_outputs),
        "tests_passed": passed,
        "status": result["status"],
        "stderr": result["stderr"],
        "compile_output": result["compile_output"],
        "details": details,
    }