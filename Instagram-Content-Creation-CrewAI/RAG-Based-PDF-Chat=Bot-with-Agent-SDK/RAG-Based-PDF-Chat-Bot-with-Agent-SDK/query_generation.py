import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from agents import SQLiteSession


async def rewrite_query_with_history(query: str, session: SQLiteSession) -> str:
    """
    Uses the LLM to rewrite a user's query into a standalone question
    based on the recent conversation history stored in SQLiteSession.
    """
    print(f"Query: {query}\n🔄 Rewriting query with context from session history...")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key == None:
         print("⚠️ WARNING: OPENAI_API_KEY not found. LLM-powered features will be skipped.")
         return None
    # OpenAI's API endpoint
    BASE_URL = "https://models.github.ai/inference"
    MODEL_NAME = "openai/gpt-4o-mini"

    try:
        llm = ChatOpenAI(
            openai_api_base=BASE_URL,
            openai_api_key=api_key,
            model=MODEL_NAME,
            # temperature=0.1,
            # max_tokens=150,
        )
        print(f"✅ LLM client initialized using {MODEL_NAME}.")
    except Exception as e:
        print(f"Error initializing LLM client: {e}")


    # 🔹 Step 1: Retrieve last 10 messages (5 user + 5 assistant)
    try:
        print("Loading session history...")
        raw_history = await session.get_items(limit=5)
        # print(f"Loaded section history: {raw_history}")
        if not raw_history:
            print("ℹ️ No chat history found in session.")
            return query
    except Exception as e:
        print(f"⚠️ Failed to load session history: {e}")
        return query

    # 🔹 Step 2: Convert to LangChain-compatible message objects
    lc_history = []
    for msg in raw_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lc_history.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_history.append(AIMessage(content=content[0]["text"]))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
            "You are an assistant that reformulates a user's new query into a complete, "
            "standalone question using the recent conversation history for context."
            "Keep the original intent of the user's question intact and simple"
            "Do not addd question details if not needed necessarily."
        ),
        ("human", 
            "Conversation history:\n{history}\n\n"
            "New question: {query}\n\n"
            "Rewrite the new question into a clear, context-rich, standalone question:"
        )
    ])

    # 🔹 Step 4: Run the LLM chain asynchronously
    chain = prompt | llm | StrOutputParser()

    try:
        print(f"History: {lc_history}")
        rewritten_query = await chain.ainvoke({"history": lc_history, "query": query})
        # rewritten_query = await chain.ainvoke({"input": query})
        rewritten_query = rewritten_query.strip() if rewritten_query else query
        print(f"🔄 Rewritten Query: {rewritten_query}")
        return rewritten_query
    except Exception as e:
        print(f"⚠️ Query rewriting failed: {e}")
        return query

# llm = get_query_llm()
# session = SQLiteSession(session_id="test", db_path="coversation_history.db")
# # llm = get_query_llm()
# query=  asyncio.run(rewrite_query_with_history(llm, "What is the paper about?", session))
