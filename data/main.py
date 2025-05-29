import os
import shutil


import config_vectordb as config
import config_model as config_model
from pdf_process_utils import load_split_pdfs
from services import get_embeddings, create_vector_store, load_vector_store, similarity_search
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


def create_index(force_recreate=False):
    """
    Create or load a vector store index from PDF files.
    
    Args:
        force_recreate (bool): If True, recreate the index even if it exists.
        
    Returns:
        FAISS: Vector store object.
    """
    if force_recreate and os.path.exists(config.VECTOR_STORE_PATH):
        shutil.rmtree(config.VECTOR_STORE_PATH)
    
    if not os.path.exists(config.VECTOR_STORE_PATH):
        print("Creating vector store index...")
        chunks = load_split_pdfs(
            pdf_dir=config.PDF_DIR_POLIZAS,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        embeddings = get_embeddings(model_name=config.EMBEDDING_MODEL)
        vector_store = create_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            vector_store_path=config.VECTOR_STORE_PATH
        )
        
        return vector_store
    else:
        print("Loading existing vector store index...")
        embeddings = get_embeddings(model_name=config.EMBEDDING_MODEL)
        return load_vector_store(vector_store_path=config.VECTOR_STORE_PATH, embeddings=embeddings)
    
def search_index(query, k=3):
    """
    Perform a similarity search on the vector store index.
    
    Args:
        query (str): Query string to search for.
        k (int): Number of top results to return.
        
    Returns:
        list: List of search results.
    """
    vector_store = load_vector_store(vector_store_path=config.VECTOR_STORE_PATH, embeddings=get_embeddings(model_name=config.EMBEDDING_MODEL))
    
    if vector_store is None:
        print("Vector store not found. Please create the index first.")
        return []
    
    similar_docs = similarity_search(vector_store=vector_store, query=query, k=k)
    for i, doc in enumerate(similar_docs):
        print(f"Result {i+1}:")
        print(f"  Document: {doc.metadata.get('source', 'Unknown')}")
        print(f"  Content: {doc.page_content[:400].replace('\n', ' ').strip()}...")
    return similar_docs


def generate_response(query, vector_store):
    """
    Generate a response using the Gemini API.
    
    Args:
        query (str): Query string to search for.

    Returns:
        str: Response from the Gemini API.
    """
    llm = ChatGoogleGenerativeAI(
        model=config_model.GEMINI_MODEL,
        google_api_key=config_model.GEMINI_API_KEY
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke(query)

    return result['result']

if __name__ == "__main__":
    # Create or load the vector store index
    vector_store = create_index(force_recreate=False)
    
    # Example search query
    type_query = input("Type 0 for search or 1 for generate response: ")
    if type_query == "0":
        query = input("Type your query: ")
        print(f"\nSearching for: {query}")
        results = search_index(query=query, k=3)
    elif type_query == "1":
        query = input("Type your query: ")
        print(f"\nGenerating response for: {query}")
        results = generate_response(query=query, vector_store=vector_store)
    else:
        print("Invalid option. Please type 0 or 1.")
        exit()

    
    if not results:
        print("No results found.")
    else:
        print(f"\nFound {len(results)} results.")
        if type_query == "0":
            for i, result in enumerate(results):
                print(f"Result {i+1}: {result.page_content[:200]}...")
        elif type_query == "1":
            print(results)