import os
from typing import List, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, function_tool, set_tracing_disabled, RunContextWrapper
from knowledgebase import PaperReranker 
from dataclasses import dataclass
from query_generation import rewrite_query_with_history

load_dotenv()
token = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
endpoint = os.getenv("OPENAI_API_BASE") or os.getenv("BASE_URL")

# Initialize client using environment variables (as done in streamlit_app.py)
client = AsyncOpenAI(base_url=endpoint, api_key=token)
set_tracing_disabled(disabled=True)

@dataclass
class pdf_name:
    name: str
    session: Any

# --- TOOL DEFINITION ---
@function_tool
async def retrieve_relevant_context(context: RunContextWrapper[pdf_name], query: str) -> str: # CHANGED to use document_name instead of state
    """
    Retrieves and reranks the most relevant chunks from the saved vector store
    based on the user query and the specified document. This is the primary tool 
    for RAG functionality.
    
    Args:
        query (str): The user's question, which may have been rewritten for context.
        document_name (str): The base name of the currently loaded PDF document. This argument MUST be sourced directly from the agent's input context/state without modification.

    Returns:
        str: The retrieved, reranked, and formatted context chunks.
    """
    # Note: We now use document_name directly as the vector store base name (Path/stem).
    base_name = context.context.name
    session = context.context.session
    print(f"📂 Agent loading vector store for: {base_name}")

    try:
        # Load existing vector store using the document_name
        reranker = PaperReranker(name=base_name)
        if not reranker.load_vector_store():
            # Updated error message since we rely on document_name
            return f"⚠️ Vector store for document '{base_name}' not found by agent. Please ensure this document is uploaded and processed."

        # Retrieve and rerank candidates
        rewritten_query =  await rewrite_query_with_history(query, session)
        candidates = reranker.retrieve_candidates(rewritten_query, top_k=8)
        reranked = reranker.rerank(rewritten_query, candidates, top_k=3)

        # Combine the most relevant content
        top_chunks = "\n\n".join(reranked["content"].tolist())
        return f"📘 Agent Retrieved Context:\n{top_chunks}"

    except Exception as e:
        # Catch and report errors, especially during model loading or FAISS operations
        return f"❌ Error retrieving context: {str(e)}"

# --- AGENT DEFINITION ---

# The Agent is defined as a regular variable, and then wrapped in a function 
# to allow for easy caching/initialization in the Streamlit app.
def get_rag_specialist_agent(client: AsyncOpenAI) -> Agent:
    """
    Initializes and returns the RAG specialist agent.
    
    Args:
        client (AsyncOpenAI): The initialized OpenAI client.
        
    Returns:
        Agent: The configured agent instance.
    """
    rag_specialist = Agent(
        name="RAG Specialist",

        instructions="""
        You are the **RAG Specialist Agent**, a highly focused and reliable information extractor for research documents. Your sole function is to provide answers based **exclusively** on the content retrieved from the loaded PDF document.

        ---
        **CORE PROTOCOL (MANDATORY)**
        
        1. **Retrieve:** You **MUST** initiate every query by calling the `retrieve_relevant_context` tool.
        2. **Context Integrity:** When calling the tool, for the `document_name` argument, you **MUST** pass the exact document identifier provided in the current state/context, without alteration, interpretation, or addition.
        3. **Analyze:** Carefully read and synthesize the information contained within the `Agent Retrieved Context` provided by the tool.
        4. **Formulate Answer:** Provide a **concise, direct, and factual** summarized answer based **ONLY** on the retrieved text. Do not use external knowledge.
        
        ---
        **RESPONSE GUIDELINES**
        
        * **No Source, No Answer:** If the retrieved context is empty, contains an error message, or clearly does not support an answer to the user's question, you **MUST** state: "I apologize, but I could not find the answer in the document provided."
        * **Format:** Your final output must be the answer itself. Do not include introductory phrases like "The answer is..." or concluding remarks like "Based on the text."
        * **Tool Error Handling:** If the tool explicitly returns a failure/error message (e.g., "Vector store for document 'X' not found"), pass that message directly to the user, prefixed with "⚠️ Document Error:".
        
        Your objective is to be an accurate, context-bound expert.
        """,
        
        # Use the gpt-4o-mini model for cost-effectiveness and speed
        model=OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=client),
        
        # The agent only uses the RAG tool
        tools=[retrieve_relevant_context] 
    )
    return rag_specialist
