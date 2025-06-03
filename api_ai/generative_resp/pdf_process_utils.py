import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_split_pdfs(pdf_dir, chunk_size, chunk_overlap):
    """
    Load and split all PDF files in a directory into chunks.

    Args:
        pdf_dir (str): Directory containing PDF files.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of document chunks.
    """
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(f"The directory {pdf_dir} does not exist.")

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the directory.")
        return []

    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
            continue

    valid_docs = [doc for doc in all_docs if isinstance(doc.page_content, str) and doc.page_content.strip()]
    if not valid_docs:
        print("No valid documents found.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(valid_docs)
    print(f"Split {len(valid_docs)} documents into {len(chunks)} chunks.")
    return chunks

def load_single_pdf(pdf_path, chunk_size, chunk_overlap):
    """
    Load and split a single PDF file into chunks.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list: List of document chunks.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return []

    valid_docs = [doc for doc in documents if isinstance(doc.page_content, str) and doc.page_content.strip()]
    if not valid_docs:
        print(f"No valid content in {pdf_path}.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(valid_docs)
    print(f"Created {len(chunks)} chunks from file {pdf_path}.")
    return chunks
