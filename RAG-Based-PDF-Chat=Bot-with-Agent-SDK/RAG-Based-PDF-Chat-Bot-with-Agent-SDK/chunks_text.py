import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_paper_text(df: pd.DataFrame, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Split extracted research paper text into smaller chunks for embedding.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['page', 'type', 'content']
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlapping characters between chunks.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['chunk_id', 'page', 'type', 'content']
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    chunks = []
    chunk_id = 0

    for _, row in df.iterrows():
        # Skip empty text
        if not isinstance(row["content"], str) or not row["content"].strip():
            continue
        if row["type"] == "NarrativeText" or row["type"] == "ListItem":
            split_texts = text_splitter.split_text(row["content"])
            import string
            printable = set(string.printable)

            for text in split_texts:            
                clean_text = ''.join(filter(lambda x: x in printable, text))
                clean_text = ' '.join(clean_text.split())
                if len(clean_text) > 50:  # Avoid very short chunks
                    chunks.append({
                        "chunk_id": chunk_id,
                        "page": row["page"],
                        "type": row["type"],
                        "content": clean_text
                    })
                    chunk_id += 1

    chunk_df = pd.DataFrame(chunks)
    return chunk_df


def preview_chunks(chunk_df: pd.DataFrame, n: int = 15):
    """Quick preview of chunked data."""
    print("\n📄 First chunks preview:")
    print(chunk_df.head(n))
    print("\n📊 Chunk type counts:")
    print(chunk_df["type"].value_counts())
    print(f"\nTotal chunks created: {len(chunk_df)}")


# # Example standalone run
# if __name__ == "__main__":
#     from partition_paper import extract_paper_elements

#     pdf_path = "pdf_path"
#     df = extract_paper_elements(pdf_path)
#     chunk_df = chunk_paper_text(df)
#     preview_chunks(chunk_df)
 