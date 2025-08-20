#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import logging
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger("qfs-evaluation")

# Determine the evaluation project's root (directory containing this script)
eval_root = Path.cwd()

# ----------------------------
# Utilities
# ----------------------------

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, capture_output: bool = True) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        env=os.environ.copy(),
    )
    out, err = proc.communicate() if capture_output else ("", "")
    return proc.returncode, out or "", err or ""


def detect_python_interpreter(qfs_root: Path) -> List[str]:
    # Prefer the library's venv if present
    candidates = [
        qfs_root / ".venv/bin/python",
        qfs_root / "venv/bin/python",
    ]
    for c in candidates:
        if c.exists():
            return [str(c)]
    # Fallbacks
    for exe in ("python3", "python"):
        code, _, _ = run_cmd([exe, "--version"], capture_output=True)
        if code == 0:
            return [exe]
    raise RuntimeError("No Python interpreter found.")


# ----------------------------
# Git helpers (operate inside QFS repo)
# ----------------------------

def git_current_branch(repo: Path) -> str:
    code, out, err = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    if code != 0:
        raise RuntimeError(f"git rev-parse failed: {err}")
    return out.strip()


def git_current_commit(repo: Path) -> str:
    code, out, err = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo)
    if code != 0:
        raise RuntimeError(f"git rev-parse HEAD failed: {err}")
    return out.strip()


def git_is_dirty(repo: Path) -> bool:
    code, out, err = run_cmd(["git", "status", "--porcelain"], cwd=repo)
    if code != 0:
        raise RuntimeError(f"git status failed: {err}")
    return bool(out.strip())


def git_stash_save(repo: Path, message: str) -> Optional[str]:
    # Save any changes; return stash ref if created, else None
    # Use -u to include untracked files
    code, out, err = run_cmd(["git", "stash", "push", "-u", "-m", message], cwd=repo)
    if code != 0:
        raise RuntimeError(f"git stash push failed: {err}")
    # If nothing to save, git prints "No local changes to save"
    if "No local changes" in (out + err):
        return None
    # Identify the stash just created by matching message
    code, out_list, err_list = run_cmd(["git", "stash", "list"], cwd=repo)
    if code != 0:
        return None
    for line in out_list.splitlines():
        if message in line:
            # Format: stash@{0}: On <branch>: <message>
            ref = line.split(":", 1)[0].strip()
            return ref
    return None


def git_fetch_branch(repo: Path, branch: str) -> None:
    # Try fetching specifically the branch
    run_cmd(["git", "fetch", "--all", "--prune"], cwd=repo)
    run_cmd(["git", "fetch", "origin", branch], cwd=repo)


def git_checkout(repo: Path, ref: str) -> None:
    code, out, err = run_cmd(["git", "checkout", ref], cwd=repo)
    if code != 0:
        raise RuntimeError(f"git checkout {ref} failed: {err}")


# ----------------------------
# QFS runner
# ----------------------------

def resolve_article_paths(inputs: List[str], base_root: Path) -> List[Path]:
    resolved: List[Path] = []
    allowed_suffixes = {".pdf", ".md", ".txt"}
    for raw in inputs:
        # Always resolve relative to eval_root
        p = (eval_root / raw).resolve()
        if p.is_dir():
            # Collect allowed files recursively
            for ext in allowed_suffixes:
                resolved.extend(p.rglob(f"*{ext}"))
        elif p.exists() and p.suffix.lower() in allowed_suffixes:
            resolved.append(p)
        else:
            raise FileNotFoundError(f"Article not found or unsupported type: {raw}")
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in resolved:
        if str(p) not in seen:
            unique.append(p)
            seen.add(str(p))
    if not unique:
        raise ValueError("No articles found from provided inputs.")
    return unique


def resolve_article_path(input_path: str, base_root: Path) -> Path:
    allowed_suffixes = {".pdf", ".md", ".txt"}
    # Always resolve relative to eval_root
    p = (eval_root / input_path).resolve()
    
    if not p.exists():
        raise FileNotFoundError(f"Article not found: {input_path}")
    if p.is_dir():
        raise ValueError(f"Expected a file for 'article', got a directory: {input_path}")
    if p.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"Unsupported article type: {input_path}")
    
    return p


def _extract_last_json_blob(text: str) -> Dict:
    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Heuristic: scan from the end, find a JSON object starting '{' with balanced braces
    braces = []
    for i, ch in enumerate(text):
        if ch == '{':
            braces.append(i)
    for start in reversed(braces):
        bal = 0
        in_str = False
        esc = False
        for j in range(start, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '{':
                    bal += 1
                elif c == '}':
                    bal -= 1
                    if bal == 0:
                        candidate = text[start:j+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break
        # Try next start
    raise json.JSONDecodeError("Could not locate a valid JSON object in output", text, 0)



def run_qfs_for_file(qfs_root: Path, py: List[str], file_path: Path, query: str, max_iterations: Optional[int], output_json_path: Path) -> Tuple[Dict, str, str]:
    cmd = py + [
        str(qfs_root / "src/main.py"),
        "--file", str(file_path),
        "--query", query,
        "--json_path", str(output_json_path),
    ]
    if max_iterations is not None:
        cmd += ["--max_iterations", str(max_iterations)]

    code, out, err = run_cmd(cmd, cwd=qfs_root)
    if code != 0:
        raise RuntimeError(f"QFS run failed for {file_path.name}: {err or out}")

    # Ensure file was written and load it (QFS now always emits JSON when --json_path supplied)
    if not output_json_path.exists():
        raise RuntimeError(f"Expected JSON output not found at {output_json_path}. Stdout snippet:\n{out[:500]}")
    try:
        data = json.loads(output_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse JSON at {output_json_path}: {e}")

    return data, out, err


def safe_slug(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "-" for c in name]
    slug = "".join(keep).strip("-")
    return slug or "output"


def append_run_csv(csv_path: Path, row: Dict[str, str]) -> None:
    headers = [
        "timestamp",
        "run_name",
        "branch",
        "commit",
        "article",
        "query",
        "max_iterations",
        "meta_json",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in headers})


def load_requests_from_json(json_path: Path) -> List[Tuple[Path, str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of {\"article\":..., \"query\":...} objects")
    
    pairs = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict) or "article" not in item or "query" not in item:
            raise ValueError(f"Entry {idx} must be an object with 'article' and 'query' fields")
        article = resolve_article_path(str(item["article"]), Path.cwd())
        query = str(item["query"])
        if not query:
            raise ValueError(f"Entry {idx} has an empty query")
        pairs.append((article, query))
        
    if not pairs:
        raise ValueError("No valid article/query pairs found in input JSON")
    return pairs


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Run batch evaluations against Query-Focused-Summarization.")
    parser.add_argument("--qfs-path", default="../Query-Focused-Summarization", help="Path to QFS repo relative to current directory")
    parser.add_argument("--branch", required=True, help="Branch in QFS repo to checkout and run")
    parser.add_argument("--run-name", required=True, help="Name for this evaluation run")
    parser.add_argument("--input-json", required=True, help="Path to requests JSON file relative to current directory")
    parser.add_argument("--output-root", default="results", help="Where to store results relative to current directory")
    parser.add_argument("--max-iterations", type=int, help="Max iterations for QFS workflow")
    parser.add_argument("--meta", help="Optional metadata as JSON string or file relative to current directory")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests to process (default: 1)")

    args = parser.parse_args()

    # Setup concise logging
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Starting run '%s' on branch '%s'", args.run_name, args.branch)

    qfs_root = (eval_root / args.qfs_path).resolve()
    logger.info("Using QFS repo at %s", qfs_root)
    if not (qfs_root.exists() and (qfs_root / "src/main.py").exists()):
        logger.error("QFS repo not found at %s", qfs_root)
        sys.exit(1)

    output_root = (eval_root / args.output_root).resolve()
    run_dir = output_root / safe_slug(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    logger.info("Results directory: %s", run_dir)

    # Parse meta
    meta_obj: Dict = {}
    if args.meta:
        meta_arg = args.meta
        meta_path = Path(meta_arg)
        try:
            if meta_path.exists():
                meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta_obj = json.loads(meta_arg)
        except Exception as e:
            logger.warning("Failed to parse meta, storing as string: %s", e)
            meta_obj = {"meta": meta_arg}

    # Persist meta.json for the run
    (run_dir / "meta.json").write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build requests (article, query) list from required JSON
    json_path = (eval_root / args.input_json).resolve()
    
    # Load requests from JSON file
    requests = load_requests_from_json(json_path)
    logger.info("Loaded %d requests from %s", len(requests), json_path)

    # Prep CSV
    csv_path = output_root / "runs.csv"

    # Git state management
    original_branch = git_current_branch(qfs_root)
    original_commit = git_current_commit(qfs_root)
    stash_ref = None
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        # Stash if dirty
        if git_is_dirty(qfs_root):
            logger.info("Local changes detected in QFS; creating stash...")
            stash_ref = git_stash_save(qfs_root, f"qfs-evaluation {timestamp} {args.run_name}")
            if stash_ref:
                logger.info("Stashed changes as %s", stash_ref)
            else:
                logger.info("No local changes to save after stash attempt")
        else:
            logger.info("No local changes in QFS repo")

        # Checkout target branch
        logger.info("Fetching and checking out branch %s", args.branch)
        git_fetch_branch(qfs_root, args.branch)
        git_checkout(qfs_root, args.branch)
        target_commit = git_current_commit(qfs_root)
        logger.info("On commit %s", target_commit[:12])

        # Choose Python
        py = detect_python_interpreter(qfs_root)
        logger.info("Using Python interpreter: %s", py[0])

        # Helper to process a single request (used for sequential + parallel)
        def _process_single(article: Path, query: str):
            qhash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
            base_slug = safe_slug(article.stem)
            # Include a short, safe snippet of the query for readability
            query_part = safe_slug(query[:40])
            name_slug = f"{base_slug}--{query_part or qhash}--{qhash}"
            qfs_json_file = run_dir / f"{name_slug}.json"
            t0 = time.time()
            data, stdout_text, stderr_text = run_qfs_for_file(qfs_root, py, article, query, args.max_iterations, qfs_json_file)
            duration = time.time() - t0
            # Save raw outputs
            raw_stdout = run_dir / "raw" / f"{name_slug}.stdout.txt"
            raw_stderr = run_dir / "raw" / f"{name_slug}.stderr.txt"
            raw_stdout.write_text(stdout_text, encoding="utf-8")
            raw_stderr.write_text(stderr_text, encoding="utf-8")
            logger.info("Finished %s (%.1fs) -> %s", article.name, duration, qfs_json_file.name)
            return {
                "timestamp": timestamp,
                "run_name": args.run_name,
                "branch": args.branch,
                "commit": target_commit,
                "article": str(article),
                "query": query,
                "max_iterations": str(args.max_iterations or ""),
                "meta_json": json.dumps(meta_obj, ensure_ascii=False),
            }

        conc = max(1, int(args.concurrency or 1))
        logger.info("Processing %d requests with concurrency=%d", len(requests), conc)

        rows_to_append: List[Dict[str, str]] = []
        if conc == 1:
            for article, query in requests:
                logger.info("Processing %s", article.name)
                rows_to_append.append(_process_single(article, query))
        else:
            # Use thread pool since work is dominated by subprocess + I/O
            with ThreadPoolExecutor(max_workers=conc) as pool:
                future_map = {pool.submit(_process_single, a, q): (a, q) for a, q in requests}
                for fut in as_completed(future_map):
                    article, query = future_map[fut]
                    try:
                        row = fut.result()
                        rows_to_append.append(row)
                    except Exception as e:
                        logger.error("Request failed for %s (%s): %s", article.name, query[:60], e)
                        raise

        # Append all CSV rows at the end to avoid interleaving writes in concurrency
        for row in rows_to_append:
            append_run_csv(csv_path, row)
        logger.info("Appended %d CSV rows", len(rows_to_append))

        # Save a manifest
        manifest = {
            "timestamp": timestamp,
            "run_name": args.run_name,
            "branch": args.branch,
            "commit": target_commit,
            "requests": [{"article": str(p), "query": q} for p, q in requests],
            "max_iterations": args.max_iterations,
            "meta": meta_obj,
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved manifest to %s", manifest_path)

        logger.info("Run complete. Results: %s | CSV: %s", run_dir, csv_path)

    finally:
        # Restore original branch; do not auto-pop stash for safety
        try:
            current = git_current_branch(qfs_root)
            if current != original_branch:
                logger.info("Restoring original branch %s", original_branch)
                git_checkout(qfs_root, original_branch)
        except Exception as e:
            logger.warning("Failed to restore original branch: %s", e)
        # Inform about stash left behind
        if stash_ref:
            logger.warning("Local changes were stashed as %s. To apply: git stash pop %s", stash_ref, shlex.quote(stash_ref))
        logger.info("Restored branch: %s @ %s", original_branch, original_commit)


if __name__ == "__main__":
    main()
