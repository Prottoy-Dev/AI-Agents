# 📄 RAG-based PDF Chatbot Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to answer questions directly from uploaded research papers (PDFs).  
It combines **LangChain**, **FAISS**, **Sentence Transformers**, and **OpenAI's agent framework** to build an interactive, document-aware chatbot interface using **Streamlit**.

---

## 🚀 Key Features

- 🔍 **Automatic PDF Parsing:** Extracts text, figures, and tables using `UnstructuredPDFLoader`.
- ✂️ **Intelligent Text Chunking:** Splits text into meaningful sections with configurable overlap.
- 🧠 **Vector Database Construction:** Builds persistent FAISS stores per document with normalized embeddings.
- 🤖 **Context-Bound Agent:** A custom RAG Specialist Agent that only answers based on retrieved context.
- 💬 **Conversation Memory:** Query rewriting with conversation history powered by SQLite.
- 🪶 **Interactive Streamlit UI:** Upload papers, ask questions, and view retrieved context chunks.

---

## 🧱 System Architecture

```text
📁 project_root/
│
├── partition_paper.py        # Extracts structured content from research PDFs
├── chunks_text.py            # Splits text into overlapping chunks for embeddings
├── knowledgebase.py          # Builds & manages FAISS vector stores and reranking
├── query_generation.py       # Rewrites user queries based on chat history
├── agent_file.py             # Defines the RAG Specialist Agent and retrieval tool
├── streamlitapp.py           # Main Streamlit app integrating all components
│
├── vector_stores/            # Auto-generated FAISS stores for uploaded documents
├── temp_pdfs/                # Temporary folder for uploaded PDFs
├── streamlit_history.db      # SQLite database for chat session persistence
└── README.md                 # (You are here)
```

---

## ⚙️ Installation
1. Clone the repository
```
git clone https://github.com/yourusername/rag-specialist-agent.git
cd rag-specialist-agent
```
2. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```
3. Install dependencies
```
pip install -r requirements.txt
```

---

## 🔑 Environment Variables
Create a .env file in the project root and include:
```
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://models.github.ai/inference
```

---

## 🧩 How It Works
### 1. PDF Extraction (partition_paper.py)

- Uses UnstructuredPDFLoader to extract elements like text, tables, and figures.

- Outputs a structured DataFrame with columns: page, type, and content.

### 2. Text Chunking (chunks_text.py)

- Cleans and splits text into overlapping chunks for embedding.

- Filters out very short or noisy segments.

### 3. Vector Store Creation (knowledgebase.py)

- Generates embeddings via SentenceTransformer.

- Builds a FAISS IndexFlatIP for cosine similarity.

- Saves both index and metadata for future reuse.

- Optionally reranks results using CrossEncoder.

### 4. Query Rewriting (query_generation.py)

- Reformulates user queries using recent chat history.

- Ensures continuity and context awareness.

- Runs asynchronously with a local SQLiteSession.

### 5. RAG Specialist Agent (agent_file.py)

- Retrieve relevant chunks using retrieve_relevant_context.

- Answer strictly based on the retrieved context.

- Avoid hallucinations or external knowledge.

### 6. Streamlit Interface (streamlitapp.py)

- Handles PDF uploads, pipeline initialization, and chat interactions.

- Displays retrieved text sources under each response.

- Automatically rebuilds vector stores for new documents.

---

## 🖥️ Usage
Run the Streamlit App
```
streamlit run streamlitapp.py
```
Steps:

- Upload a PDF — The system extracts, chunks, and builds a FAISS index.

- Ask Questions — The RAG agent retrieves relevant context and answers based only on the document.

- View Sources — The top retrieved chunks with similarity and rerank scores appear below each answer.

---

## 🧰 Example Workflow
| Step | Description                          | Module                 |
| ---- | ------------------------------------ | ---------------------- |
| 1️⃣  | Upload research paper PDF            | `streamlitapp.py`      |
| 2️⃣  | Extract elements from PDF            | `partition_paper.py`   |
| 3️⃣  | Split into text chunks               | `chunks_text.py`       |
| 4️⃣  | Build FAISS index                    | `knowledgebase.py`     |
| 5️⃣  | Rewrite query using history          | `query_generation.py`  |
| 6️⃣  | Retrieve and rerank relevant context | `agent_file.py`        |
| 7️⃣  | Generate final LLM answer            | `RAG Specialist Agent` |

---

## 🧩 Example Output

User Query:
```
What method was used for feature extraction?
```
Agent Response:
```
The paper used a convolutional neural network (CNN) architecture to extract features from the metal surface images.
```
Retrieved Chunks:
```
Source 1 (Page 5)
Type: NarrativeText
Content: "The convolutional neural network (CNN) architecture was employed to extract high-level visual features from surface defect images..."
```

---

## 🧠 Notes

- Vector stores are cached under vector_stores/<document_name>/ for reuse.

- Each chat session is stored in streamlit_history.db (SQLite).

- Works offline after embedding & FAISS creation (only query rewriting and answering use OpenAI API).

- Supports GPU acceleration for both embedding and reranking if CUDA is available.

---

## 🔍 Room for Future Improvements

- Multi-document retrieval and comparison

- Improved partitioning/chunking via the Unstructured API
 
- Better chunking and retrieval strategy with summarization and citation linking with paid multimodal LLMs
