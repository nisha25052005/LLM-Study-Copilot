from typing import List, Dict
from .pdf_utils import extract_text_from_pdf, chunk_text
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .llm import LocalLLM


class RAGPipeline:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = None
        self.chunks: List[Dict] = []
        self.llm = LocalLLM()  # load LLM once

    # ---------- Ingestion ----------

    def ingest_pdfs(self, uploaded_files) -> None:
        all_chunks = []
        for f in uploaded_files:
            pages = extract_text_from_pdf(f)
            chunks = chunk_text(pages)
            for c in chunks:
                c["source"] = f.name
            all_chunks.extend(chunks)

        self.chunks = all_chunks

        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedding_model.encode(texts)

        self.vector_store = VectorStore(dim=embeddings.shape[1])
        metadata = [
            {
                "chunk_id": c["chunk_id"],
                "page_num": c["page_num"],
                "text": c["text"],
                "source": c["source"],
            }
            for c in all_chunks
        ]
        self.vector_store.add_embeddings(embeddings, metadata)

    # ---------- Retrieval ----------

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.vector_store is None:
            return []
        query_emb = self.embedding_model.encode([query])[0]
        results = self.vector_store.search(query_emb, top_k=top_k)
        return results

    def build_context_string(self, query: str, top_k: int = 5, max_chars: int = 3500) -> str:
        """
        Build a single big context string from top chunks, trimmed to max_chars.
        """
        results = self.retrieve_context(query, top_k=top_k)
        if not results:
            return ""

        parts = []
        total = 0
        for r in results:
            text = r["text"].replace("\n", " ")
            tag = f"[{r['source']} - p.{r['page_num']}] "
            chunk_str = tag + text
            if total + len(chunk_str) > max_chars:
                break
            parts.append(chunk_str)
            total += len(chunk_str)

        return "\n\n".join(parts)

    # ---------- Old dummy answer (still keep if needed) ----------

    def answer_question_dummy(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve_context(query, top_k=top_k)
        if not results:
            return "No documents indexed yet or no relevant context found."

        context_strs = []
        for r in results:
            snippet = r["text"][:300].replace("\n", " ")
            context_strs.append(
                f"[{r['source']} - p.{r['page_num']}] {snippet}..."
            )

        joined = "\n\n".join(context_strs)
        answer = (
            "Top relevant context from your notes:\n\n"
            f"{joined}\n\n"
            "👉 This is raw context. Switch to LLM mode for a proper answer."
        )
        return answer

    # ---------- LLM-Powered Q&A ----------

    def answer_question_llm(self, query: str, top_k: int = 5) -> str:
        if self.vector_store is None:
            return "Please upload and index PDFs first."

        context = self.build_context_string(query, top_k=top_k)
        if not context:
            return "I couldn't find relevant content in your notes for this question."

        prompt = f"""
You are a helpful study assistant for a student. You must answer ONLY using the given context from their notes.

Context:
{context}

Question: {query}

Instructions:
- Answer in clear, simple language.
- Keep the answer within 4–8 lines.
- If you use any specific info, mention the page in square brackets like [p. 3].
- If the answer is not in the context, say you cannot find it in the notes.
Answer:
"""
        return self.llm.generate(prompt, max_new_tokens=256)

    # ---------- Global context for summaries/quizzes ----------

    def build_global_context(self, max_chars: int = 4000) -> str:
        """
        Use first few chunks from all PDFs to build a big context for summary/quiz.
        """
        if not self.chunks:
            return ""

        parts = []
        total = 0
        for c in self.chunks:
            text = c["text"].replace("\n", " ")
            tag = f"[{c['source']} - p.{c['page_num']}] "
            chunk_str = tag + text
            if total + len(chunk_str) > max_chars:
                break
            parts.append(chunk_str)
            total += len(chunk_str)

        return "\n\n".join(parts)

    # ---------- Study helpers: Summary, MCQs, Flashcards ----------

    def generate_summary(self) -> str:
        context = self.build_global_context()
        if not context:
            return "Please upload and index PDFs first."

        prompt = f"""
You are a study assistant. Summarize the following notes for exam revision.

Context:
{context}

Instructions:
- Give a structured summary with bullet points.
- Highlight key definitions, formulas, or concepts.
- Keep it concise but meaningful (around 10–15 bullet points).
Summary:
"""
        return self.llm.generate(prompt, max_new_tokens=400)

    def generate_mcqs(self, num_questions: int = 8) -> str:
        context = self.build_global_context()
        if not context:
            return "Please upload and index PDFs first."

        prompt = f"""
You are a question paper setter. Create {num_questions} multiple-choice questions (MCQs) from the student's notes.

Context:
{context}

Instructions:
- Each question must have 4 options (A, B, C, D).
- Mark the correct answer clearly after each question.
- Cover different topics from the notes.
MCQs:
"""
        return self.llm.generate(prompt, max_new_tokens=600)

    def generate_flashcards(self, num_cards: int = 12) -> str:
        context = self.build_global_context()
        if not context:
            return "Please upload and index PDFs first."

        prompt = f"""
You are creating flashcards for revision. From the notes below, create {num_cards} Q–A style flashcards.

Context:
{context}

Instructions:
- Format as: Q: ...  A: ...
- Each question should test one core concept.
- Keep answers short (1–3 lines).
Flashcards:
"""
        return self.llm.generate(prompt, max_new_tokens=600)
