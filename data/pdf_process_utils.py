import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_split_pdfs(pdf_dir, chunk_size, chunk_overlap):
    """
    Load and split PDF files from a directory into chunks.
    
    Args:
        pdf_dir (str): Directory containing PDF files.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap size between chunks.
        
    Returns:
        list: List of document chunks.
    """
    # Load PDFs from the specified directory
    loader = PyPDFLoader(pdf_dir)
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"The directory {pdf_dir} does not exist.")
    try:
        documents = loader.load()
    except Exception as e:
        print(f"Failed to load PDFs from {pdf_dir}: {e}")

    valid_docs = []
    for i, doc in enumerate(documents):
        if not isinstance(doc.page_content, str):
            print(f"Document {i}, source {doc.metadata.get('source', 'N/A')} has invalid content type: {type(doc.page_content)}. Skipping.")
            continue
        if not doc.page_content.strip():
            print(f"Document {i}, source {doc.metadata.get('source', 'N/A')} is empty. Skipping.")
            continue
        valid_docs.append(doc)
    
    if not valid_docs:
        print("No valid documents found in the specified directory.")
        return []
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    print(f"Splitting {len(valid_docs)} documents into chunks of size {chunk_size} with overlap {chunk_overlap}.")
    try:
        chunks = text_splitter.split_documents(valid_docs)
    except Exception as e:
        print(f"Failed to split documents: {e}")
        return []
    
    if not chunks:
        print("No chunks were created from the documents.")
        return []
    print(f"Created {len(chunks)} chunks from the documents.")
    
    return chunks