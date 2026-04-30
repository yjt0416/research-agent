# Demo Script

## Goal

Use this script to record a 2-3 minute demo that shows the project as a practical research assistant instead of a generic chat app.

## Recording Flow

1. Start on the homepage.
   Say: "This is Research Agent Copilot, a LangGraph and FastAPI based research agent for document-grounded Q&A, guarded tool use, and reproducible technical workflows."

2. Upload a TXT, PDF, or DOCX file.
   Say: "After a document is uploaded, the backend parses it, chunks it, embeds it, and stores it in Chroma for grounded retrieval."

3. Ask a grounded question.
   Suggested prompt: `Please summarize the core idea and cite the relevant chunks.`
   Say: "The answer is generated with retrieval context, and the response includes source references instead of a pure model guess."

4. Trigger a Python task.
   Suggested prompt: `Please run this Python code and print the result of sum([10, 20, 30]).`
   Say: "Potentially sensitive tool actions go through an approval step, so the agent stays useful without becoming unsafe."

5. Run the paper reproduction workflow.
   Suggested prompt: `Please reproduce this paper with runnable Python code and generate a technical report.`
   Say: "In research mode, the agent plans the task, retrieves the paper context, generates a standalone script, executes it, and returns downloadable code, figures, metrics, and a Markdown report."

6. Download an artifact.
   Say: "The output is not just chat text. Users can directly download the generated Markdown report, Python reproduction script, or the ZIP bundle."

## Recording Tips

- Keep the browser zoom at 100 percent so the interface feels clean and readable.
- Use one prepared sample document for grounded Q&A and one prepared paper path for reproduction.
- Keep the terminal ready in the background in case you want to show FastAPI logs for credibility.
- If the reproduction run takes longer, mention that the workflow is generating files and evaluating a paper-specific experiment.

## Recommended Ending

Close with:

"This project focuses on research-heavy workflows where grounded retrieval, controllable tools, and downloadable outputs matter more than chat alone."
