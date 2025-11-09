import streamlit as st
import os
import asyncio
from typing import Tuple, List, Any, Dict
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI 
from agent_file import get_rag_specialist_agent, pdf_name 
from agents import SQLiteSession, Runner
from partition_paper import extract_paper_elements 
from chunks_text import chunk_paper_text           
from knowledgebase import PaperReranker   
from query_generation import rewrite_query_with_history        

# --- 0. ENVIRONMENT AND INITIALIZATION ---
load_dotenv()
token = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
endpoint = os.getenv("OPENAI_API_BASE") or os.getenv("BASE_URL")

if not token:
    st.error("⚠️ OPENAI_API_KEY or API_KEY not found in environment variables. Please set it to enable LLM features.")
    st.stop()


# --- 1. CORE PIPELINE FUNCTIONS ---

def load_and_store_pdf(uploaded_file) -> str:
    """
    Implements the core document processing: extract, chunk, and build/save the vector store.
    """
    # 1. Save PDF temporarily
    temp_dir = Path("temp_pdfs")
    temp_dir.mkdir(exist_ok=True)
    pdf_path = temp_dir / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Document name is the file's stem (filename without extension)
    document_name = Path(uploaded_file.name).stem

    # 2. Extract elements, chunk text, and build vector store
    st.info(f"Extracting elements from '{uploaded_file.name}'...")
    paper_df = extract_paper_elements(str(pdf_path))
    chunk_df = chunk_paper_text(paper_df)

    # 3. Build Vector Store
    st.info(f"Building vector store for document name: '{document_name}'...")
    reranker = PaperReranker(name=document_name)
    reranker.build_vector_store(chunk_df)

    return document_name #, chunk_df


@st.cache_resource
def build_rag_pipeline() -> Tuple[AsyncOpenAI, Any]: 
    """
    Initializes and caches the AsyncOpenAI client and the RAG specialist agent.
    
    Returns:
        Tuple[AsyncOpenAI, Agent]: (openai_client, rag_agent)
    """
    # 1. Initialize AsyncOpenAI client (required by the Agent)
    try:
        client = AsyncOpenAI(base_url=endpoint, api_key=token)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None, None
        
    # 2. Initialize the RAG agent using the client
    rag_agent = get_rag_specialist_agent(client=client)
    
    st.info("✅ Agent and OpenAI client initialized.")
    return client, rag_agent


# --- AGENT-BASED PIPELINE FUNCTION ---

async def run_agent_pipeline(
    rag_agent: Any, # Agent instance
    doc_id_for_rag: str,
    query: str, 
    session: SQLiteSession,
    # chunk_df: Any
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes the full RAG pipeline flow by running the RAG Specialist agent,
    ensuring history persistence using Runner.run().
    """
    # 1. Perform Query Rewrite for conversation context
    with st.spinner("Rewriting query for context..."):
        # Pass session object to the query rewriter
        # rewritten_query = await rewrite_query_with_history(query_llm, query, session)
        rewritten_query = await rewrite_query_with_history(query, session)

    # st.warning("Skipping query rewriting due to missing LLM key.")
    
    # st.markdown(f"_Agent's Input Query: {rewritten_query}_", help="The query passed to the RAG Specialist Agent after history context was applied.")
        
    # 2. Run the Agent using Runner.run() and pass the session object
    with st.spinner("RAG Specialist Agent is generating answer..."):
        # CHANGED: Use Runner.run() to automatically manage the session history
        # The prompt is the rewritten query.
        result = await Runner.run(
            rag_agent, 
            query, 
            session= session, # NEW: Pass the session here for history persistence
            context= pdf_name(name=doc_id_for_rag, session=session) # Pass required state for the tool
        )
        final_answer = result.final_output # Extract the final answer from the result

    # 3. Simulate Source Retrieval for UI Display (using the rewritten query)
    try:
        reranker = PaperReranker(name=doc_id_for_rag)
        # chunks = chunk_df
        if reranker.load_vector_store():
            candidates = reranker.retrieve_candidates(rewritten_query, top_k=8)
            reranked_df = reranker.rerank(rewritten_query, candidates, top_k=3)
            # print(reranked_df)
            top_chunks = reranked_df.to_dict('records')
            print(top_chunks)
        else:
            top_chunks = [{"content": "Could not reload vector store to display sources.", "page": "N/A"}]
    except Exception as e:
        top_chunks = [{"content": f"Source simulation failed: {str(e)}", "page": "N/A"}]

    return final_answer, top_chunks


# --- 2. STREAMLIT APPLICATION ---

async def streamlit_app():
    """The main Streamlit application function."""
    st.set_page_config(page_title="RAG Specialist Q&A", layout="wide")
    st.title("📄 RAG Specialist Agent Q&A")

    # --- Session State Setup ---
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    # 1. Session Management
    st.header("Session Management")
    session_id = st.text_input("Session ID:", value="default-session", help="Enter a unique ID to save/load your chat history.")
    
    # Initialize SQLiteSession
    try:
        session = SQLiteSession(session_id=session_id, db_path="streamlit_history.db")
    except NameError:
        st.error("SQLiteSession is not defined. Ensure the 'agents' module is accessible.")
        return

    # Initialize RAG Pipeline components (Agent and Client)
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = build_rag_pipeline()
    
    # Unpack components
    openai_client, rag_agent = st.session_state.rag_pipeline

    if rag_agent is None:
        st.error("Failed to initialize RAG Agent components. Check API keys and console for errors.")
        return


    # --- Document Upload and Processing (load_and_store_pdf integration) ---
    st.header("1. Document Setup")
    uploaded_file = st.file_uploader(
        "Upload a PDF Research Paper", 
        type="pdf", 
        disabled=st.session_state.is_processing,
        key="pdf_uploader",
        help="The file name (stem) will be used as the document identifier."
    )

    process_button = st.button("Rebuild RAG Pipeline", key="process_btn", type="primary", disabled=st.session_state.is_processing or not uploaded_file)
    
    if uploaded_file and (not st.session_state.document_name or process_button):
        st.session_state.is_processing = True
        
        with st.spinner(f"Loading and processing {uploaded_file.name}..."):
            document_name= load_and_store_pdf(uploaded_file)
            # st.session_state.chunk_df = chunk_df
            st.session_state.document_name = document_name
        
        st.success(f"✅ Document '{document_name}' processed and ready!")
        st.session_state.is_processing = False
        await session.clear_session() # Clear history for the new document
        st.rerun()

    if st.session_state.document_name:
        st.success(f"**Active Document:** `{st.session_state.document_name}`")
    else:
        st.warning("No document processed. Please upload a PDF and click 'Rebuild RAG Pipeline' (or wait for automatic processing after upload) to begin.")
        return 


    # --- Chat Interface (run_agent_pipeline integration) ---
    st.header("2. Ask a Question")
    
    # FIX: Use asyncio.run() to await the asynchronous get_history() method
    try:
        # Retrieve chat messages from history retrieved from session
        chat_history = await session.get_items()
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        chat_history = []

    for message in chat_history:
        # print(message)
        # Ensure the 'content' field is accessed correctly, handling the SDK's nested structure
        if message.get("role") == "user":
            content_text = message.get("content", "")
            with st.chat_message("user"):
                # st.write(content_text)
                st.markdown(content_text)
        elif message.get("role") == "assistant":
            content_text = message.get("content", "")[0].get("text")
            with st.chat_message("assistant"):
                st.markdown(content_text)
                # st.write(content_text)

    query = st.chat_input("Enter your query about the document...")

    if query and st.session_state.document_name:
        print(f"Document Name in Session State: {st.session_state.document_name}")
        # 1. We no longer manually add messages here, as Runner.run() handles it.
        # We only display the user query immediately.
        with st.chat_message("user"):
            st.markdown(query)

        # 2. Run RAG Pipeline (Agent)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Call the agent-based function, passing the session object
            # The session will be updated inside run_agent_pipeline by Runner.run()
            # answer, sources = asyncio.run(
            #     run_agent_pipeline(
            #         rag_agent=rag_agent,
            #         document_name=st.session_state.document_name,
            #         query=query, 
            #         session=session 
            #     )
            # )
            answer, sources = await run_agent_pipeline(
                    rag_agent=rag_agent,
                    doc_id_for_rag=st.session_state.document_name,
                    query=query, 
                    session=session,
                    # chunk_df=st.session_state.chunk_df 
                )
            message_placeholder.markdown(answer)
            st.session_state["latest_sources"] = sources
            st.rerun()

    # --- After rerun, persist and display retrieved chunks ---
    if "latest_sources" in st.session_state:
        st.subheader("Top Retrieved Chunks")
        for idx, source in enumerate(st.session_state["latest_sources"]):
            st.markdown(
                f"**Source {idx+1} (Page {source.get('page', 'N/A')})** — "
                f"Score: {source.get('score', 0):.2f}, Rerank: {source.get('rerank_score', 0):.2f}"
            )
            st.caption(f"Type: {source.get('type', 'Unknown')} | Chunk ID: {source.get('chunk_id', 'N/A')}")
            st.code(source.get('content', 'Content not available'))
            # st.write("---")
        
        # Rerun to show the newly added assistant message from the session
        # st.rerun()

# Run the application
if __name__ == "__main__":
    asyncio.run(streamlit_app())




# import streamlit as st
# def load_and_store_pdf(pdf_path):

# def build_rag_pipeline(vector_db):

# def test_rag_pipeline(rag_pipeline, query):

# def streamlit_app():
#     st.title("RAG Specialist Chatbot")
#     st.write("This is a placeholder for the RAG Specialist Streamlit application.") 
#     pdf_path = "vector_stores"
#     if "vector_db" not in st.session_state or st.button("Reload Document"):
#         with st.spinner("Loading and processing PDF..."):
#             st.session_state.store = load_and_store_pdf(pdf_path)
   
#     if "rag_pipeline" not in st.session_state or st.button("Rebuild RAG Pipeline"):
#         with st.spinner("Building RAG pipeline..."):
#             st.session_state.rag_pipeline = build_rag_pipeline(st.session_vector_db)
    
#     query = st.text_input("Enter your query about the document:")
#     if query:
#         answer ,sources = test_rag_pipeline(st.session_state.rag_pipeline, query)
#         st.subheader("Answer:")
#         st.write(answer)

#         st.subheader("Top Sources:")
#         for idx, source in enumerate(sources):
#             st.markdown(f"Source {idx+1}: **")
#             st.write(source.page_content)
