---
name: review-python-tests
description: Review existing Python tests for missing cases and weak assertions. Load only when the user asks for a Python test review or coverage assessment.
---

# Review Python tests

1. Find the relevant Python test modules and compare their assertions and fixtures with the behavior of the application code. Read the files. Do not change any test files or production code as part of the review.
2. Check the expected behavior, error handling, boundary inputs and whether each assertion proves the promised behavior. Look for fragile fixtures or state shared across tests. Cite the test file and test name or line for each concrete finding.
3. If it helps and the operator permits it, run the existing relevant suite with the supported `python -m unittest` command. Report what actually ran. Base any remaining observations on inspection.
4. Return the most important findings first and suggest the exact missing cases. If there are no Python tests or no substantive findings, state that directly.
