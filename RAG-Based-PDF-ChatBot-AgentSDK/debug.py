# from partition_paper import *
# import pandas as pd
# from langchain_community.document_loaders import UnstructuredPDFLoader
# loader = UnstructuredPDFLoader("pdf_path", mode="elements")
# docs = loader.load()
# records = []
# for doc in docs:
#     content = doc.page_content.strip()
#     # Skip garbage or too-short elements
#     if len(content) < 25 or content.isascii() is False:
#         continue
#     records.append({
#         "page": doc.metadata.get("page_number", None),
#         "type": doc.metadata.get("category", "text"),
#         "content": doc.page_content
#     })

# # print(records[0:10])
# # for doc in docs:
# #     print(doc.metadata.get("category", "text"))
# df = pd.DataFrame(records)

# from chunks_text import *
# chunk_df = chunk_paper_text(df) 
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"},
#     encode_kwargs={
#         "normalize_embeddings": True,
#         "convert_to_numpy": True,
#         "batch_size": 8
#         }
#     )
# texts = chunk_df["content"].astype(str).tolist()
# try:
#     batch_size = 8
#     embeddings_lists = []
    
#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]
#         embeddings_batch = embeddings.embed_documents(batch)
#         embeddings_lists.extend(embeddings_batch)

#     vector_db = FAISS.from_embeddings(
#         embedding=embeddings,
#         text_embeddings=list(zip(texts, embeddings_lists))
#     )
# except Exception as e:
#     print(f"Error during embedding: {e}")
#     simple_texts = [' '.join(t.split()) for t in texts]
#     vector_db = FAISS.from_texts(simple_texts, embeddings)

# retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
# query = "What is this paper about?"
# results = retriever.get_relevant_documents(query)
# print(f"Top retrieved documents for query: '{query}'\n")
# for i, doc in enumerate(results):
#     print(f"--- Document {i+1} ---")
#     print(doc.page_content)
#     print("\n")