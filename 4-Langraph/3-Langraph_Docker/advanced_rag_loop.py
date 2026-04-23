"""
Advanced RAG with LangGraph
---------------------------
This example demonstrates:
1) Multi-node workflow
2) Retrieval quality loop
3) Answer quality loop
4) Source-aware final responses

Run:
    python advanced_rag_loop.py
"""

import os
from typing import Dict, List, Literal, TypedDict

import numpy as np
import ollama
from langgraph.graph import END, StateGraph


class State(TypedDict):
    question: str
    rewritten_question: str
    chunks: List[Dict]
    retrieved: List[Dict]
    context_blocks: List[Dict]
    answer: str
    retrieval_score: float
    retrieval_attempt: int
    generation_attempt: int
    need_retrieval_retry: bool
    need_answer_retry: bool
    debug_notes: List[str]


class KnowledgeStore:
    """Loads local docs, creates embeddings, and performs semantic search."""

    def __init__(self) -> None:
        self.chunks: List[Dict] = []

    def load_docs(self, file_names: List[str]) -> List[Dict]:
        self.chunks = []
        print("\n[LOAD] Reading project documents...")

        for file_name in file_names:
            if not os.path.exists(file_name):
                print(f"  - Missing file: {file_name}")
                continue

            with open(file_name, "r", encoding="utf-8") as f:
                text = f.read()

            parts = self._split_text(text, chunk_size=260)
            print(f"  - {file_name}: {len(parts)} chunks")

            for i, part in enumerate(parts):
                self.chunks.append(
                    {
                        "chunk_id": f"{file_name}#{i}",
                        "source": file_name,
                        "text": part,
                        "embedding": self._embed(part),
                    }
                )

        print(f"[LOAD] Total chunks indexed: {len(self.chunks)}")
        return self.chunks

    def _split_text(self, text: str, chunk_size: int = 260) -> List[str]:
        text = " ".join(text.split())
        output: List[str] = []
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size].strip()
            if piece:
                output.append(piece)
        return output

    def _embed(self, text: str) -> List[float]:
        try:
            res = ollama.embed(model="nomic-embed-text", input=text)
            return res["embeddings"][0]
        except Exception as ex:
            print(f"  ! Embedding fallback used: {ex}")
            return [0.0] * 768

    def search(self, query: str, top_k: int = 3, exclude_ids: List[str] | None = None) -> List[Dict]:
        exclude = set(exclude_ids or [])
        query_embedding = self._embed(query)
        scored: List[Dict] = []

        for chunk in self.chunks:
            if chunk["chunk_id"] in exclude:
                continue
            similarity = self._cosine(query_embedding, chunk["embedding"])
            scored.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "similarity": similarity,
                }
            )

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def _cosine(self, vec1: List[float], vec2: List[float]) -> float:
        a = np.array(vec1)
        b = np.array(vec2)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)


store = KnowledgeStore()
store_ready = False


def node_load(state: State) -> State:
    global store_ready
    print("\n[1] LOAD_NODE")

    if not store_ready:
        files = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt"]
        store.load_docs(files)
        store_ready = True
    else:
        print("  - Reusing already indexed chunks.")

    return {
        **state,
        "chunks": store.chunks,
        "retrieved": [],
        "context_blocks": [],
        "answer": "",
        "retrieval_score": 0.0,
        "retrieval_attempt": 1,
        "generation_attempt": 1,
        "need_retrieval_retry": False,
        "need_answer_retry": False,
        "debug_notes": [],
        "rewritten_question": state["question"],
    }


def node_rewrite_question(state: State) -> State:
    print("\n[2] REWRITE_QUESTION_NODE")
    base_prompt = (
        "Rewrite the user question into a short retrieval-friendly query. "
        "Keep intent and key nouns. Output only rewritten query."
    )
    q = state["question"]
    try:
        res = ollama.chat(
            model="gemma3:latest",
            messages=[{"role": "user", "content": f"{base_prompt}\nQuestion: {q}"}],
            options={"temperature": 0.1, "num_predict": 40},
        )
        rewritten = res["message"]["content"].strip()
    except Exception:
        rewritten = q

    print(f"  - Original : {q}")
    print(f"  - Rewritten: {rewritten}")
    return {**state, "rewritten_question": rewritten}


def node_retrieve(state: State) -> State:
    print("\n[3] RETRIEVE_NODE")
    used_ids = [d["chunk_id"] for d in state["retrieved"]]
    exclude = used_ids if state["retrieval_attempt"] > 1 else []
    top_chunks = store.search(state["rewritten_question"], top_k=3, exclude_ids=exclude)

    for i, c in enumerate(top_chunks, 1):
        print(f"  - {i}. {c['source']} score={c['similarity']:.3f}")

    combined = state["retrieved"] + top_chunks
    avg = sum(c["similarity"] for c in top_chunks) / len(top_chunks) if top_chunks else 0.0
    return {**state, "retrieved": combined, "retrieval_score": avg}


def node_grade_retrieval(state: State) -> State:
    print("\n[4] GRADE_RETRIEVAL_NODE")
    score = state["retrieval_score"]
    attempt = state["retrieval_attempt"]
    threshold = 0.55

    retry = score < threshold and attempt < 3
    note = f"retrieval_attempt={attempt}, score={score:.3f}, retry={retry}"
    print(f"  - {note}")

    return {
        **state,
        "need_retrieval_retry": retry,
        "debug_notes": state["debug_notes"] + [note],
    }


def node_improve_retrieval(state: State) -> State:
    print("\n[5] IMPROVE_RETRIEVAL_NODE")
    new_attempt = state["retrieval_attempt"] + 1
    print(f"  - Increment retrieval attempt -> {new_attempt}")
    return {**state, "retrieval_attempt": new_attempt, "need_retrieval_retry": False}


def node_build_context(state: State) -> State:
    print("\n[6] BUILD_CONTEXT_NODE")
    ranked = sorted(state["retrieved"], key=lambda x: x["similarity"], reverse=True)
    unique = []
    seen = set()
    for r in ranked:
        if r["chunk_id"] in seen:
            continue
        seen.add(r["chunk_id"])
        unique.append(r)
        if len(unique) == 5:
            break
    print(f"  - Context blocks selected: {len(unique)}")
    return {**state, "context_blocks": unique}


def _build_generation_prompt(state: State, strict: bool) -> str:
    context_text = ""
    for i, c in enumerate(state["context_blocks"], 1):
        context_text += (
            f"[{i}] Source={c['source']} Score={c['similarity']:.3f}\n"
            f"{c['text']}\n\n"
        )

    strict_line = (
        "If uncertain, answer exactly: I cannot answer this from the provided documents."
        if strict
        else "If not found, clearly say the docs do not contain it."
    )

    return (
        "You are a document-grounded assistant.\n"
        "Use only the context.\n"
        f"{strict_line}\n"
        "Keep answer concise and factual.\n\n"
        f"QUESTION:\n{state['question']}\n\n"
        f"CONTEXT:\n{context_text}\n"
    )


def node_generate_answer(state: State) -> State:
    print("\n[7] GENERATE_ANSWER_NODE")
    strict = state["generation_attempt"] > 1
    prompt = _build_generation_prompt(state, strict=strict)

    if not state["context_blocks"]:
        answer = "I cannot answer this from the provided documents."
    else:
        try:
            res = ollama.chat(
                model="gemma3:latest",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 220},
            )
            answer = res["message"]["content"].strip()
        except Exception:
            res = ollama.chat(
                model="gemma:2b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2, "num_predict": 220},
            )
            answer = res["message"]["content"].strip()

    return {**state, "answer": answer}


def node_validate_answer(state: State) -> State:
    print("\n[8] VALIDATE_ANSWER_NODE")
    answer = state["answer"].lower()
    context_blob = " ".join(c["text"].lower() for c in state["context_blocks"])
    attempt = state["generation_attempt"]

    # Lightweight grounding check:
    # if answer has very little lexical overlap, retry generation once.
    answer_tokens = [t for t in answer.replace("\n", " ").split(" ") if len(t) > 4]
    overlap = sum(1 for t in answer_tokens if t in context_blob)
    overlap_ratio = (overlap / len(answer_tokens)) if answer_tokens else 1.0

    retry = overlap_ratio < 0.20 and attempt < 2
    note = f"generation_attempt={attempt}, overlap_ratio={overlap_ratio:.2f}, retry={retry}"
    print(f"  - {note}")

    return {
        **state,
        "need_answer_retry": retry,
        "debug_notes": state["debug_notes"] + [note],
    }


def node_improve_answer(state: State) -> State:
    print("\n[9] IMPROVE_ANSWER_NODE")
    new_attempt = state["generation_attempt"] + 1
    print(f"  - Increment generation attempt -> {new_attempt}")
    return {**state, "generation_attempt": new_attempt, "need_answer_retry": False}


def route_after_retrieval_grade(state: State) -> Literal["improve_retrieval", "build_context"]:
    if state["need_retrieval_retry"]:
        print("  -> Routing: improve_retrieval (loop)")
        return "improve_retrieval"
    print("  -> Routing: build_context")
    return "build_context"


def route_after_answer_validation(state: State) -> Literal["improve_answer", "done"]:
    if state["need_answer_retry"]:
        print("  -> Routing: improve_answer (loop)")
        return "improve_answer"
    print("  -> Routing: done")
    return "done"


def build_graph():
    graph = StateGraph(State)

    graph.add_node("load", node_load)
    graph.add_node("rewrite_question", node_rewrite_question)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_retrieval", node_grade_retrieval)
    graph.add_node("improve_retrieval", node_improve_retrieval)
    graph.add_node("build_context", node_build_context)
    graph.add_node("generate_answer", node_generate_answer)
    graph.add_node("validate_answer", node_validate_answer)
    graph.add_node("improve_answer", node_improve_answer)

    graph.set_entry_point("load")
    graph.add_edge("load", "rewrite_question")
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("retrieve", "grade_retrieval")

    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_retrieval_grade,
        {"improve_retrieval": "improve_retrieval", "build_context": "build_context"},
    )

    graph.add_edge("improve_retrieval", "retrieve")
    graph.add_edge("build_context", "generate_answer")
    graph.add_edge("generate_answer", "validate_answer")

    graph.add_conditional_edges(
        "validate_answer",
        route_after_answer_validation,
        {"improve_answer": "improve_answer", "done": END},
    )

    graph.add_edge("improve_answer", "generate_answer")
    return graph.compile()


def print_sources(result: State) -> None:
    print("\nSources used:")
    ranked = sorted(result["context_blocks"], key=lambda x: x["similarity"], reverse=True)
    for i, c in enumerate(ranked, 1):
        snippet = c["text"][:130].strip()
        print(f"{i}. {c['source']} score={c['similarity']:.3f}")
        print(f"   {snippet}...")


def print_debug(result: State) -> None:
    print("\nLoop / quality stats:")
    print(f"- Retrieval attempts : {result['retrieval_attempt']}")
    print(f"- Generation attempts: {result['generation_attempt']}")
    print(f"- Final retrieval score: {result['retrieval_score']:.3f}")
    if result["debug_notes"]:
        print("- Trace:")
        for note in result["debug_notes"]:
            print(f"  * {note}")


def main():
    print("=" * 70)
    print("ADVANCED RAG WITH MULTI-NODE LOOPS (LangGraph + Ollama)")
    print("=" * 70)
    print("Knowledge files: doc1.txt, doc2.txt, doc3.txt, doc4.txt")
    print("Commands: 'quit', 'sources', 'stats'")
    print("-" * 70)

    graph = build_graph()
    last_result: State | None = None

    while True:
        q = input("\nYou: ").strip()

        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        if q.lower() == "sources":
            if last_result:
                print_sources(last_result)
            else:
                print("Ask a question first.")
            continue

        if q.lower() == "stats":
            if last_result:
                print_debug(last_result)
            else:
                print("Ask a question first.")
            continue

        if not q:
            continue

        initial: State = {
            "question": q,
            "rewritten_question": q,
            "chunks": [],
            "retrieved": [],
            "context_blocks": [],
            "answer": "",
            "retrieval_score": 0.0,
            "retrieval_attempt": 1,
            "generation_attempt": 1,
            "need_retrieval_retry": False,
            "need_answer_retry": False,
            "debug_notes": [],
        }

        try:
            result = graph.invoke(initial)
            last_result = result
            print("\nAssistant:")
            print(result["answer"])
        except Exception as ex:
            print(f"\nError: {ex}")
            print("Check Ollama service and models:")
            print("  ollama pull nomic-embed-text")
            print("  ollama pull gemma3:latest")
            print("  ollama pull gemma:2b")


if __name__ == "__main__":
    main()
