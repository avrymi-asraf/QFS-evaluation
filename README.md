# QFS-evaluation

## Purpose

This project provides a batch evaluation tool for the Query-Focused-Summarization (QFS) system. It automates running the QFS tool against multiple articles and queries, across different code versions (Git branches). The primary goal is to enable systematic testing and comparison of summarization outputs.

## Structure

The project is centered around the `main.py` script, which orchestrates the evaluation workflow. It interacts with a sibling `Query-Focused-Summarization` Git repository to run different versions of the summarization tool. The evaluator handles Git operations (stashing local changes, checking out branches), invokes the QFS command-line interface for each input, and saves the structured JSON output and logs.

### File Structure

- `main.py`: The main evaluation script.
- `ui`: A simple web interface to view the results.
- `requests/requests.sample.json`: An example of the input JSON file format.
- `results/`: The default directory where all evaluation outputs are stored, organized by run name.
- `README.md`: This file.

### Special Files

The evaluation process is driven by an input JSON file specified with the `--input-json` argument. This file contains an array of objects, where each object defines an `article` to be processed and a `query` for summarization.

Example `requests/requests.sample.json`:
```json
[
  {
    "article": "articles/Attention-Is-All-You-Need.pdf",
    "query": "What are the core innovations of the Transformer architecture and how do they replace recurrence?"
  },
  {
    "article": "articles/The Linear Representation Hypothesis.pdf",
    "query": "Summarize the linear representation hypothesis and its implications for neural network interpretability."
  }
]
```

For each article and query pair, the evaluation generates detailed JSON output in the results directory. The output includes:
- A comprehensive summary
- A set of automatically generated QA pairs to validate the summary
- Multiple iterations of refinement if needed
- Metadata about the evaluation run

## Usage

### Quick Start

1.  **Prerequisites:**
    *   Ensure you have a sibling directory named `Query-Focused-Summarization` containing the QFS project.
    *   The QFS project should have its dependencies installed (either in a `.venv` or `venv` directory).
    *   Set up the required environment variables for the LLM service.

2.  **Run an evaluation:**

    ```bash
    python main.py --branch <your-qfs-branch> --run-name <your-run-name> --input-json requests/requests.sample.json
    ```

    *   `--branch`: The Git branch in the QFS repository to evaluate.
    *   `--run-name`: A unique name for this evaluation run. Outputs will be saved in `results/<run-name>/`.
    *   `--input-json`: Path to the JSON file with articles and queries. Each run will create output files in the results directory, preserving the article's name but with a .json extension.

### Other Usage Modes

You can customize the evaluation with additional arguments:

*   **Add metadata to a run:**
    ```bash
    python main.py --branch my-branch --run-name exp-002 --input-json ./requests.json --meta '{"seed": 1, "notes": "baseline"}'
    ```

*   **Limit the number of articles to process:**
    ```bash
    python main.py --branch my-branch --run-name exp-003 --input-json ./requests.json --max-iterations 3
    ```

*   **Specify custom paths for the QFS repo and output directory:**
    ```bash
    python main.py --branch my-branch --run-name exp-004 --input-json ./requests.json --qfs-path ../My-QFS-Fork --output-root ./custom-results
    ```

*   **Process requests concurrently (parallel execution):**
  ```bash
  python main.py --branch my-branch --run-name exp-005 --input-json ./requests.json --concurrency 4
  ```
  Set `--concurrency` to the number of simultaneous article/query executions you want. Each output file name now includes a short hash and snippet of the query to avoid collisions when the same article appears with different queries. Example pattern:
  ```
  <article-stem>--<query-snippet>--<hash>.json
  ```
  Raw stdout/stderr for each run are stored under `results/<run-name>/raw/` with the same slug and `.stdout.txt` / `.stderr.txt` suffixes.
