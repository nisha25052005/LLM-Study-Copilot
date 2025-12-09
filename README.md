# 📚 LLM Study Copilot (RAG-Based AI Study Assistant)

The **LLM Study Copilot** is a Retrieval-Augmented Generation (**RAG**)–powered study tool that turns uploaded PDF notes into interactive learning material.
It can **answer questions**, **summarize notes**, **generate MCQs**, and **create flashcards** using a **local LLM (Qwen2.5-0.5B-Instruct)** — no API keys or paid services needed.

Built with **Python, Streamlit, FAISS, Sentence Transformers**, and **HuggingFace Transformers**.

---

#  Features

###  1. Ask Questions From Your PDF Notes (RAG)

* Upload multiple PDFs
* Extracts + chunks text
* Creates embeddings using Sentence Transformers
* Uses FAISS for semantic search
* Retrieves relevant chunks and answers using your notes only
* Page numbers included

###  2. LLM-Powered Answers (Local Offline Qwen Model)

* Runs **Qwen2.5-0.5B-Instruct** locally
* Provides contextual, grounded answers
* No API keys → No cost → Full privacy

###  3. Automatic Summaries

* 10–15 bullet-point summaries
* Key definitions, formulas, concepts

###  4. MCQ Generator

* Generates exam-style MCQs
* Options A–D
* Correct answer shown

###  5. Flashcard Creator

* Produces concise Q–A flashcards
* Perfect for quick revision

###  6. Streamlit UI

Simple interface with:

* PDF Upload
* Knowledge Base Builder
* Ask Questions
* Generate Summary / MCQs / Flashcards

---

#  Project Architecture

PDF → Extract Text → Chunk → Embed → Store in FAISS
Query → Embed → Retrieve Relevant Chunks → Build Prompt → Qwen Answer

This ensures **low hallucination**, **fast retrieval**, and **accurate context-based answers**.

---

#  Tech Stack

**Frontend:** Streamlit
**Backend:** Python
**Embeddings:** Sentence Transformers (MiniLM-L6-v2)
**Vector Store:** FAISS
**LLM:** Qwen2.5-0.5B-Instruct (HuggingFace)
**PDF Processing:** PyPDF2
**Frameworks:** Transformers, Accelerate

---

#  Folder Structure

study-copilot/
│── app.py
│── backend/
│   ├── pdf_utils.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── llm.py
│── requirements.txt
│── README.md
│── venv/ (ignored)

---

#  How to Replicate This Project (Step-by-Step)

Anyone can run this project locally by following the instructions below.

---

##  **1. Clone the Repository**

```
git clone https://github.com/YOUR_USERNAME/study-copilot.git
cd study-copilot
```

Replace `YOUR_USERNAME` with your GitHub username.

---

##  **2. Create a Virtual Environment**

```
python -m venv venv
```

Activate it:

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

---

##  **3. Install All Dependencies**

If you have `requirements.txt`:

```
pip install -r requirements.txt
```

If not, install manually:

```
pip install streamlit PyPDF2 sentence-transformers faiss-cpu transformers accelerate torch sentencepiece safetensors numpy python-dotenv
```

---

##  **4. Download the LLM Automatically**

The first time you run the project, Hugging Face will automatically download:

```
Qwen/Qwen2.5-0.5B-Instruct
```

No API key needed.

---

##  **5. Run the Application**

```
streamlit run app.py
```

You should see:

```
Local URL: http://localhost:8501
```

Open it in your browser.

---

##  **6. Use the App**

1. Upload one or more PDFs
2. Click **Build Knowledge Base**
3. Ask any question
4. Or generate:

   * Summary
   * MCQs
   * Flashcards

Enjoy! 
Everything runs **fully offline**.

---

# requirements.txt (For Replication)

Include this in your repo:

```
streamlit
PyPDF2
sentence-transformers
faiss-cpu
transformers
accelerate
torch
sentencepiece
safetensors
numpy
python-dotenv
```

---

# Future Enhancements

* Support DOCX / PPT / Image OCR
* Multiple model selector (Qwen, TinyLlama, Mistral, Phi)
* Export MCQs/Flashcards to PDF
* User authentication + cloud storage
* UI improvements

---

#  Credits

* Hugging Face (Transformers + Qwen model)
* FAISS
* Streamlit
* Sentence Transformers


