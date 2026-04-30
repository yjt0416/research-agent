# Resume Bullets

## Chinese Version

- 设计并实现基于 FastAPI、LangChain 与 LangGraph 的科研 Agent 系统，支持文档解析、RAG 检索、任务拆解、工具调用、受控 Python 执行与可下载结果产物生成。
- 构建本地资料检索增强链路，完成 TXT/PDF/DOCX 解析、文本切分、语义向量检索、Top-k 召回与引用溯源问答，提升科研资料问答的准确性与可解释性。
- 基于 LangGraph StateGraph 拆分 Intent Router、Planner、Retriever、Tool Router、Code Executor、Reflector、Reporter 等节点，实现复杂任务的条件路由、失败重试与反思修正。
- 设计论文复现工作流，可根据本地论文生成可运行 Python 脚本、实验图表、指标 JSON、Markdown 技术文档与 ZIP 打包结果，支持科研场景下的实验复现与结果交付。

## English Version

- Built a full-stack research agent with FastAPI, LangChain, and LangGraph for document parsing, RAG, task planning, guarded tool use, restricted Python execution, and downloadable output artifacts.
- Implemented a local-document retrieval pipeline covering TXT, PDF, and DOCX ingestion, text chunking, semantic vector search, top-k recall, and source-grounded question answering.
- Modeled the agent as a LangGraph StateGraph with router, planner, retriever, tool router, code executor, reflector, and reporter nodes to support conditional routing, retries, and self-correction.
- Added a paper reproduction workflow that generates runnable Python, figures, metrics JSON, Markdown technical reports, and ZIP bundles from local research papers.
