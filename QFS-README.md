# Query-Focused Summarization Copilot Instructions

## Project Overview
This is a query-focused document summarization system using iterative AI agents. The system processes articles (PDF/text) and generates summaries tailored to specific user queries through a multi-agent workflow.

## Architecture & Agent Workflow
The core workflow in `src/main.py` orchestrates 4 specialized agents in sequence:

1. **QuestionGenerator** - Creates diagnostic questions from query + article
2. **Summarizer** - Generates summaries, can highlight specific sections on subsequent iterations
3. **QAAgent** - Answers questions based solely on the current summary
4. **Judge** - Compares summary against full article, identifies missing topics

This creates an iterative refinement loop where the Judge's feedback drives the next iteration's focus areas.

## Key Implementation Patterns

### Agent Design (src/Agents.py)
- All agents use **LangChain Expression Language (LCEL)** chains: `prompt | llm | parser`
- Shared global LLM instance: `ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")`
- Custom output parsers for structured data: `QuestionListParser`, `QAPairsParser`, `JudgeOutputParser`
- Each agent takes specific inputs and returns typed outputs (List[str], List[Tuple[str, str]], etc.)

### File Processing
- PDF support via dual loader strategy: `PyPDFLoader` → fallback to `UnstructuredPDFLoader`
- Automatic markdown conversion with page headers and content cleaning
- Text files read directly with UTF-8 encoding

### Command Line Interface
Critical arguments for `src/main.py`:
- `--file`: Article path (PDF or text)
- `--query`: Query string for focused summarization
- `--max_iterations`: Iteration limit (default: 5)
- `--output_format`: `print` (console) or `json` (structured output)

### Environment Setup
- Requires `GOOGLE_API_KEY` environment variable (load via python-dotenv)
- Virtual environment activation: `source .venv/bin/activate`
- Dependencies: LangChain ecosystem, Google Generative AI, PDF processing libs

## Development Workflow

### Testing & Running
```bash
# Quick demo with existing article
./run_demo.sh

# Custom run
python src/main.py --file articals/attention-is-all-you-need.pdf --query "attention mechanisms" --max_iterations 3 --output_format json
```

### Adding New Agents
1. Inherit from base pattern: `__init__(llm=None)` with global fallback
2. Define LCEL chain: `prompt | llm | output_parser`
3. Implement `run()` method with typed inputs/outputs
4. Add custom output parser if needed in same file

### JSON Output Structure
When using `--output_format json`, the system returns:
- `query`, `max_iterations`, `total_iterations`, `status`
- `iterations[]` array with per-iteration data: `summary`, `qa_pairs`, `needs_iteration`, `missing_topics`
- `final_summary`

## Project Conventions
- Source code in `src/` directory
- Example articles in `articals/` (note the spelling)
- Single shared LLM instance for cost efficiency
- Error handling with graceful fallbacks (especially PDF loading)
- Clean separation: main.py (orchestration) vs Agents.py (agent logic)

