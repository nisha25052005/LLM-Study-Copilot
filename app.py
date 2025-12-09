import streamlit as st
from backend.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Study Copilot", layout="wide")

# Keep a single pipeline instance in session
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

rag = st.session_state.rag

st.title("📚 LLM Study Copilot (RAG)")

# ---------- Sidebar: Upload & Index ----------

st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload your notes/slides/articles (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("🔄 Build Knowledge Base") and uploaded_files:
    with st.spinner("Processing PDFs and building vector index..."):
        rag.ingest_pdfs(uploaded_files)
    st.sidebar.success("Indexed successfully! You can start asking questions.")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Upload lecture slides, textbooks, or PDF notes.")

# ---------- Q&A Section ----------

st.subheader("❓ Ask Questions from Your Notes")

col_mode1, col_mode2 = st.columns(2)
with col_mode1:
    answer_mode = st.selectbox(
        "Answer Mode",
        ["Show retrieved context only", "Use LLM to generate answer"],
        index=1
    )

query = st.text_input("Type your question here")
ask_btn = st.button("Ask")

if ask_btn:
    if not uploaded_files:
        st.warning("Please upload and index at least one PDF first.")
    elif not query.strip():
        st.warning("Please type a question.")
    else:
        if answer_mode == "Show retrieved context only":
            with st.spinner("Retrieving context..."):
                answer = rag.answer_question_dummy(query)
            st.text_area("Retrieved Context", answer, height=300)
        else:
            with st.spinner("Retrieving context and asking LLM..."):
                answer = rag.answer_question_llm(query)
            st.markdown("#### 🧠 LLM Answer")
            st.write(answer)

# ---------- Study Helpers: Summary, MCQs, Flashcards ----------

st.markdown("---")
st.subheader("📌 Study Helpers")

col1, col2, col3 = st.columns(3)

summary_text = mcq_text = flash_text = None

with col1:
    if st.button("📝 Generate Summary"):
        with st.spinner("Generating summary from your notes..."):
            summary_text = rag.generate_summary()

with col2:
    if st.button("❓ Generate MCQs"):
        with st.spinner("Creating MCQs..."):
            mcq_text = rag.generate_mcqs()

with col3:
    if st.button("🎴 Generate Flashcards"):
        with st.spinner("Creating flashcards..."):
            flash_text = rag.generate_flashcards()

if summary_text:
    st.markdown("### 📝 Summary")
    st.text_area("Summary", summary_text, height=300)

if mcq_text:
    st.markdown("### ❓ MCQs")
    st.text_area("MCQs", mcq_text, height=350)

if flash_text:
    st.markdown("### 🎴 Flashcards")
    st.text_area("Flashcards", flash_text, height=350)
