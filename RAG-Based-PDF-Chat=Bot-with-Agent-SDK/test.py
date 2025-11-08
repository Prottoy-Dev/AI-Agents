from agents import Runner,  SQLiteSession 
from partition_paper import *
from query_generation import *
from knowledgebase import *
from chunks_text import *
from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
# from api import SessionWrapper  # Or define SessionWrapper inline if needed

# session = SQLiteSession(session_id="1234", db_path="history.db")
# new_items = [
#     {"role": "user", "content": "pdf name"}
# ]
# session.add_items(new_items)
# async def main():
#     # 1. Initiate your session (specifying the file path)
#     session = SQLiteSession(session_id="1234", db_path="history.db")
    
#     new_items = [
#         {"role": "user", "content": "pdf name"}
#     ]
    
#     # 2. **CRITICAL:** Use the 'await' keyword
#     await session.add_items(new_items)
    
#     print("Items added successfully. Checking for history.db...")

# # 3. **CRITICAL:** Run the async function
# asyncio.run(main())
# print("🧠 Running agent with SQLiteSession...\n")

# pdf_path = r"F:\Courses\Ostad\Assignments\Capstone\1706.03762v7.pdf"
# base_name = Path(pdf_path).stem
# chunks = partition_pdf(
#     filename=pdf_path,

#     infer_table_structure=True,             # extract tables
#     strategy="hi_res",                      # mandatory to infer tables

#     # extract_image_block_types=["image"], 
#     # image_output_dir_path="output_path",  # directory to save images 

       
#     # extract_image_block_to_payload=False,    # if True, extract base64 images for API usage   

#     chunking_strategy="by_title",
#     max_characters=1000,
#     combine_text_under_n_chars=200,
#     new_after_n_chars=600,
#     )
# chunk_df = pd.DataFrame(chunks)
# print("\n📄 First chunks preview:")
# print(chunk_df.head(10))
# # print(chunks[0:10])
# print(f"\nTotal chunks created: {len(chunk_df)}")
# for x in range(10):
#     print(f"LOADING CHUNK NUMBER {x}:\n")
#     print(chunks[x])
# print("Loading and chunking document...")
# print(f"Chunks={chunks[5].to_dict()}")
# chunks_dicts = [chunk for chunk in chunks]
# print(f"Total chunks created: {len(chunks_dicts)}")
# print("Sample chunk content:")
# print(chunks_dicts[1])  # Print first 500 characters of the first chunk

# df = extract_paper_elements(pdf_path)
# preview_paper_data(df)

# df = chunk_paper_text(df)

# for index,row in df.iterrows():
#     if row['type'] == 'CompositeElement':
#         print(f"CompositeElement  Text (Page {row['page']}):\n{row['content']}\n")

# # print(f"Uncategorized Chunks:\n{chunk_df[chunk_df['type']=='Uncategorized']}")
# preview_chunks(df)
# print("Chunk ID:", df.loc[2, "chunk_id"])
# print("Page:", df.loc[2, "page"])
# print("Type:", df.loc[2, "type"])
# print("Content:\n", df.loc[2, "content"])

# reranker = PaperReranker(name=base_name)
# reranker.build_vector_store(chunk_df)
# # reranker = PaperReranker(name=base_name)
# reranker.load_vector_store()
# QUERY = "What is self attention?"
# cands = reranker.retrieve_candidates(QUERY, top_k=5)
# reranked = reranker.rerank(QUERY, cands, 3)
# print(f"Reranked Results:\n{reranked}")
# print(reranked[["content"]].head(3))
# rewritten_query = asyncio.run(rewrite_query_with_history(
#     llm=get_query_llm(),
#     query=user_query,
#     session=session
# ))
# candidates = pr.retrieve_candidates(rewritten_query, 20)
# print("Candidates:\n", candidates[["content", "score"]])
# reranked = pr.rerank(rewritten_query, candidates, 10)
# context_text = "\n\n---\n\n".join(reranked["content"].tolist())
# print("Context for LLM:\n", context_text)
# async def test_agent():
#     # Create a SQLite-based session (in-memory)
#     # session = SQLiteSession(session_id="console-test")
#     print("🧠 Running agent with SQLiteSession...\n")
#     initialize_rag_knowledge_base(r"F:\Courses\Ostad\Assignments\Capstone\1706.03762v7.pdf")
#     user_query = input("Enter your query: ")
#     rewritten_query = await rewrite_query_with_history(
#         llm=get_query_llm(),
#         query=user_query,
#         session=session
#     )
#     print(f"Rewritten Query: {rewritten_query}")
#     agent = main_agent

#     # 1️⃣ Basic question
#     response1 = await Runner.run(
#         agent,
#         input=rewritten_query,
#         session=session
#     )
#     print("Response 1:", response1.final_output)

#     # # 2️⃣ Follow-up question (context should persist)
#     # response2 = await Runner.run(
#     #     agent,
#     #     input="Who did I say I am?",
#     #     session=session
#     # )
#     # print("\nResponse 2:", response2.final_output)

#     # Close the session to clean up
#     session.close()

# if __name__ == "__main__":
#     asyncio.run(test_agent())
    # print(session.memory.load_messages())

# session = Session(session_id="test-session-123")
# # Safe wrapper for Runner.run (copied from your api.py)
# async def test_agent():
#     print("🧠 Running agent with built-in Session...\n")

#     # 1. Simple test
#     response = await Runner.run(
#         main_agent,
#         query="Hello, what is diabetes?",
#         session=session
#     )
#     print("Response 1:", response.final_output)

#     # 2. Follow-up test (shows session memory works)
#     response2 = await Runner.run(
#         main_agent,
#         query="Can you list common symptoms?",
#         session=session
#     )
#     print("\nResponse 2:", response2.final_output)

# if __name__ == "__main__":
#     asyncio.run(test_agent())

# async def run_with_session(agent, user_query: str, session: Session) -> str:
#     result = await Runner.run(starting_agent=agent, input=user_query, session=session)

#     return result.final_output if hasattr(result, 'final_output') else str(result)
#     # Await only if the result is awaitable
#     # if inspect.isawaitable(result):
#     #     result = await result

#     # # Return the final_output if it exists, otherwise string representation
#     # try:
#     #     return result.final_output
#     # except AttributeError:
#     #     return str(result)

# async def test_agent():
#     # Create a session
#     # session = SessionWrapper(history=[], session_id="test-session-123")
#     user_query = "Hello, how are you?"

#     # Run the agent
#     output = await run_with_session(main_agent, user_query, session)

#     print("Agent Output:", output)
#     print("Session History:", [(msg.role, msg.content) for msg in session.history])


# if __name__ == "__main__":
#     asyncio.run(test_agent())

# async def test_agent():
#     print("🏥 Testing Agent...")
    
#     # Test 1: General medical question
#     print("\n1. Testing knowledge:")
#     response1 = await Runner.run(main_agent, "What are the common treatments for diabetes?")
#     print("Response:", response1.final_output)
    
#     # Test 2: Database query
#     print("\n2. Testing database query:")
#     response2 = await Runner.run(main_agent, "How many patients are in the heart disease dataset?")
#     print("Response:", response2.final_output)

#     # Test 3: Database query
#     print("\n3. Testing database query:")
#     response3 = await Runner.run(main_agent, "My age is 58, I am male, I have no smoking history and I have previous cancer history. What is my risk of cancer?")
#     print("Response:", response3.final_output)

#     print("\n✅ Agent is working successfully!")

# # Run the test
# if __name__ == "__main__":
#     asyncio.run(test_agent())
#     # run_interactive(main_agent, client)