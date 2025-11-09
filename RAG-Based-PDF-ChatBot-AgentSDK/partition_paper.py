import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader

def extract_paper_elements(pdf_path: str):
    """
    Extract structured elements (text, tables, figures) from a research paper PDF.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        pd.DataFrame: A DataFrame with columns [page, type, content].
    """
    loader = UnstructuredPDFLoader(
        pdf_path, 
        mode="elements",
        strategy="hi_res", 
        # chunking_strategy="by_title"
        )
    docs = loader.load()

    records = []
    for doc in docs:
        records.append({
            "page": doc.metadata.get("page_number", None),
            "type": doc.metadata.get("category", "text"),
            "content": doc.page_content
        })

    df = pd.DataFrame(records)
    return df


def preview_paper_data(df: pd.DataFrame, n: int = 10):
    """Simple helper to preview extracted data and type counts."""
    print("\n📄 Preview of extracted content:")
    print(df.head(n))
    print("\n📊 Element type counts:")
    print(df["type"].value_counts())
