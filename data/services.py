import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings(model_name, device='cpu'):
    """
    Get HuggingFace embeddings for the specified model.
    
    Args:
        model_name (str): Name of the HuggingFace model.
        device (str): Device to use for embeddings ('cpu' or 'cuda').
        
    Returns:
        HuggingFaceEmbeddings: Embeddings object.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"Successfully loaded embeddings model: {model_name} on {device}.")
        return embeddings
    except Exception as e:
        print(f"Error loading embeddings model {model_name}: {e}")
        return None
    
def create_vector_store(chunks, embeddings, vector_store_path):
    '''
    Create a vector store from document chunks using the specified embeddings.
    Args:
        chunks (list): List of document chunks.
        embeddings (HuggingFaceEmbeddings): Embeddings object.
        vector_store_path (str): Path to save the vector store.
    Returns:
        FAISS: Vector store object.
    '''
    try:
        if not chunks:
            print("No chunks provided to create vector store.")
            return None
        if not embeddings:
            print("Embeddings model is not provided.")
            return None
        
        vector_store = FAISS.from_documents(documents=chunks, embedding= embeddings)
        if not os.path.exists(os.path.dirname(vector_store_path)):
            os.makedirs(os.path.dirname(vector_store_path))
        vector_store.save_local(vector_store_path)

        print(f"Vector store created and saved at {vector_store_path}.")
        return vector_store
    
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None
    
def load_vector_store(vector_store_path, embeddings):
    """
    Load a vector store from the specified path.
    
    Args:
        vector_store_path (str): Path to the vector store.
        embeddings (HuggingFaceEmbeddings): Embeddings object.
        
    Returns:
        FAISS: Loaded vector store object.
    """
    try:
        if not os.path.exists(vector_store_path):
            print(f"Vector store path {vector_store_path} does not exist.")
            return None
        
        vector_store = FAISS.load_local(
            folder_path=vector_store_path,
            embeddings= embeddings,
            allow_dangerous_deserialization= True)
        
        print(f"Vector store loaded from {vector_store_path}.")
        return vector_store
    
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None
    
def similarity_search(vector_store, query, k=3):
    """
    Perform a similarity search on the vector store.
    
    Args:
        vector_store (FAISS): Vector store object.
        query (str): Query string for similarity search.
        k (int): Number of results to return.
        
    Returns:
        list: List of similar documents.
    """
    try:
        if not vector_store:
            print("Vector store is not provided.")
            return []
        
        results = vector_store.similarity_search(query=query, k=k)
        print(f"Found {len(results)} similar documents for query: '{query}'.")
        return results
    
    except Exception as e:
        print(f"Error during similarity search: {e}")
        return []