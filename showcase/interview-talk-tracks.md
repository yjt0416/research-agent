# Interview Talk Tracks

## 30-Second Version

I built a full-stack research agent called Research Agent Copilot using FastAPI, LangChain, and LangGraph. It supports document parsing, RAG over local files, guarded tool calling, restricted Python execution, and downloadable outputs such as Markdown reports, Python scripts, and ZIP bundles. I also added a research workflow that can ingest a paper, plan the task, reproduce an experiment in code, and generate a technical report.

## 1-Minute Version

Research Agent Copilot is a practical AI assistant for paper reading, technical document Q&A, and lightweight experiment reproduction. On the backend I used FastAPI for APIs, LangChain for model integration, and LangGraph StateGraph to split the workflow into router, planner, retriever, tool router, code executor, reflector, and reporter nodes. For grounding, uploaded TXT, PDF, and DOCX files are parsed, chunked, embedded with sentence-transformers, and stored in Chroma. I also built a guarded tool layer so file reads and Python execution can require approval. The project goes beyond chat because it can return downloadable Markdown, Python, JSON, image, and ZIP artifacts.

## 3-Minute Version

This project started from a simple goal: make research-oriented AI workflows feel as easy as chat, but still keep them grounded, inspectable, and safe. I designed the system around three layers.

The first layer is the user interface and API layer. The frontend is a lightweight chat interface, and the backend exposes FastAPI endpoints for standard chat, document upload, path-based ingestion, agent workflows, tool execution, artifact download, and evaluation.

The second layer is the agent orchestration layer. I used LangGraph StateGraph to model the workflow as a set of explicit nodes: Intent Router, Planner, Retriever, Tool Router, Code Executor, Reflector, and Reporter. That structure makes the system easier to debug than a monolithic chain, and it lets me do conditional routing, retries, and reflection instead of relying on one prompt.

The third layer is the knowledge and execution layer. Documents are parsed and chunked, semantic embeddings are generated with sentence-transformers, and the chunks are stored in Chroma for retrieval-augmented answering. For execution, I added restricted Python and file tools, plus a human approval gate for higher-risk actions. On top of that, I built a paper reproduction mode that can ingest a local paper, retrieve experimental details, generate a standalone Python script, execute it, and package the results into downloadable artifacts.

The result is a project that demonstrates not just prompt calling, but system design across retrieval, orchestration, tool use, and output delivery.

## Likely Follow-Up Questions

### Why use LangGraph instead of only LangChain?

LangChain is still used for model and prompt composition, but LangGraph is a better fit for explicit multi-step stateful workflows. It gives clearer control over routing, retries, and node-by-node reasoning.

### How do you reduce hallucination?

I use RAG over local files, force source-aware answers for grounded tasks, and keep tool usage explicit. For paper reproduction, the report also states the engineering assumptions instead of pretending every extracted parameter is exact.

### What is the most practical part of the system?

The artifact pipeline. Instead of only returning text in chat, the system can produce real deliverables such as Markdown reports, Python scripts, metrics JSON, figures, and ZIP bundles.
