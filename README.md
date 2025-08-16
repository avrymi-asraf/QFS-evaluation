# QFS-evaluation

Batch evaluator for the sibling Query-Focused-Summarization (QFS) repo.

What it does
- Stashes local changes in QFS (if any), checks out a target branch, and runs the QFS CLI across per-article requests.
- Saves one JSON output per article under results/<run-name>/ plus manifest.json and meta.json.
- Appends a CSV log (results/runs.csv) with branch, commit, run name, article, query, and run metadata.
- Restores your original branch when finished and leaves a stash reference if one was created.

Assumptions
- Folder layout: this project and the QFS repo are siblings under the same parent directory.
  - Default QFS path: ../Query-Focused-Summarization
  - Override with --qfs-path if needed.

Requirements
- QFS repo has its dependencies installed (preferably a .venv inside QFS). The runner will use QFS/.venv/bin/python if found, else fall back to python3/python.
- Environment variable GOOGLE_API_KEY must be set for the QFS LLM.

Input format (required)
- Provide --input-json pointing to a JSON array of objects with per-article queries:

Example `examples/requests.sample.json`:
```
[
  {
    "article": "articals/Attention-Is-All-You-Need.pdf",
    "query": "What are the core innovations of the Transformer architecture and how do they replace recurrence?"
  },
  {
    "article": "articals/The Linear Representation Hypothesis.pdf",
    "query": "Summarize the linear representation hypothesis and its implications for neural network interpretability."
  }
]
```

Usage (examples)
- Minimal:
  python QFS-evaluation/main.py --branch my-branch --run-name exp-001 --input-json QFS-evaluation/examples/requests.sample.json
- With metadata:
  python QFS-evaluation/main.py --branch my-branch --run-name exp-002 --input-json ./requests.json --meta '{"seed": 1, "notes": "baseline"}'
- Control iterations:
  python QFS-evaluation/main.py --branch my-branch --run-name exp-003 --input-json ./requests.json --max-iterations 3
- Custom locations:
  python QFS-evaluation/main.py --branch my-branch --run-name exp-004 --input-json ./requests.json --qfs-path ../Query-Focused-Summarization --output-root ./results

Output structure
- results/<run-name>/
  - <article-name>.json  (structured JSON from QFS --output_format json)
  - manifest.json        (summary of run parameters)
  - meta.json            (the metadata object you passed)
- results/runs.csv       (appended per-article rows)

Notes
- The runner tolerates extra prints in QFS stdout and extracts the last JSON object.
- If local changes exist in QFS, they are stashed with a unique message; the stash is not auto-applied at the end.