# Advanced LangGraph RAG (With Multiple Loops)

This folder contains a practical "advanced" RAG example built with LangGraph.

## What this RAG does

The assistant answers questions about a fictional transit modernization program called **CityTransit**.  
It only uses these files:

- `doc1.txt` (project outcomes and forecasting)
- `doc2.txt` (privacy and governance)
- `doc3.txt` (architecture and reliability)
- `doc4.txt` (rollout, budget, risks)

If the answer is not in these documents, the assistant should say it cannot answer.

---

## Workflow design

`advanced_rag_loop.py` uses multiple nodes and two loops:

1. `load`  
   Loads and embeds document chunks once.
2. `rewrite_question`  
   Rewrites user query to improve retrieval.
3. `retrieve`  
   Runs semantic search.
4. `grade_retrieval`  
   Checks retrieval quality score.
5. `improve_retrieval`  
   **Loop 1**: retries retrieval (up to 3 attempts).
6. `build_context`  
   Builds top context set for generation.
7. `generate_answer`  
   Produces grounded answer.
8. `validate_answer`  
   Checks if answer appears grounded in context.
9. `improve_answer`  
   **Loop 2**: retries answer generation in stricter mode.

So graph behavior is:

`load -> rewrite -> retrieve -> grade -> (loop or continue) -> build_context -> generate -> validate -> (loop or END)`

---

## How to run

1. Start Ollama.
2. Pull models:

```bash
ollama pull nomic-embed-text
ollama pull gemma3:latest
ollama pull gemma:2b
```

3. Run:

```bash
python advanced_rag_loop.py
```

---

## Interactive commands

Inside chat:

- Ask any question related to CityTransit docs.
- `sources` -> see retrieved source chunks.
- `stats` -> see loop attempts and quality notes.
- `quit` -> exit.

---

## How to build this yourself (step-by-step learning plan)

1. **Start with simple RAG**
   - Chunk docs
   - Embed docs
   - Retrieve top-k
   - Generate answer from context
2. **Move to LangGraph**
   - Put each step in a node
   - Pass shared `State`
3. **Add retrieval quality loop**
   - Compute average similarity
   - If score low, retry retrieval
4. **Add answer quality loop**
   - Validate grounding
   - Retry with stricter prompt
5. **Add transparency**
   - Print sources and loop stats

This progression is exactly what production-grade agentic pipelines do: retrieve, evaluate, improve, then answer.
