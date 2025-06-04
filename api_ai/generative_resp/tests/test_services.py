import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
from langchain.schema import Document

# Import the functions to test
# Assuming your original script is saved as 'services.py'
from services import get_embeddings, create_vector_store, load_vector_store, similarity_search


class TestGetEmbeddings:
    """Test cases for get_embeddings function"""
    
    @patch('services.HuggingFaceEmbeddings')
    def test_successful_embedding_creation(self, mock_embeddings):
        """Test successful embedding model creation"""
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        
        with patch('builtins.print') as mock_print:
            result = get_embeddings("sentence-transformers/all-MiniLM-L6-v2", "cpu")
        
        assert result == mock_embedding_instance
        mock_embeddings.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
        mock_print.assert_called_with("Successfully loaded embeddings model: sentence-transformers/all-MiniLM-L6-v2 on cpu.")
    
    @patch('services.HuggingFaceEmbeddings')
    def test_embedding_creation_with_cuda(self, mock_embeddings):
        """Test embedding creation with CUDA device"""
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        
        with patch('builtins.print') as mock_print:
            result = get_embeddings("sentence-transformers/all-MiniLM-L6-v2", "cuda")
        
        assert result == mock_embedding_instance
        mock_print.assert_called_with("Successfully loaded embeddings model: sentence-transformers/all-MiniLM-L6-v2 on cuda.")
    
    @patch('services.HuggingFaceEmbeddings')
    def test_embedding_creation_exception(self, mock_embeddings):
        """Test handling of exceptions during embedding creation"""
        mock_embeddings.side_effect = Exception("Model not found")
        
        with patch('builtins.print') as mock_print:
            result = get_embeddings("invalid-model", "cpu")
        
        assert result is None
        mock_print.assert_called_with("Error loading embeddings model invalid-model: Model not found")
    
    @patch('services.HuggingFaceEmbeddings')
    def test_default_device_parameter(self, mock_embeddings):
        """Test that default device parameter is 'cpu'"""
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        
        with patch('builtins.print'):
            result = get_embeddings("test-model")
        
        assert result == mock_embedding_instance


class TestCreateVectorStore:
    """Test cases for create_vector_store function"""
    
    @patch('services.FAISS')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.path.dirname')
    def test_successful_vector_store_creation(self, mock_dirname, mock_makedirs, mock_exists, mock_faiss):
        """Test successful vector store creation"""
        # Setup mocks
        mock_chunks = [MagicMock(), MagicMock()]
        mock_embeddings = MagicMock()
        mock_vector_store = MagicMock()
        
        mock_faiss.from_documents.return_value = mock_vector_store
        mock_dirname.return_value = "/test/path"
        mock_exists.return_value = True
        
        with patch('builtins.print') as mock_print:
            result = create_vector_store(mock_chunks, mock_embeddings, "/test/path/vector_store")
        
        assert result == mock_vector_store
        mock_faiss.from_documents.assert_called_once_with(documents=mock_chunks, embedding=mock_embeddings)
        mock_vector_store.save_local.assert_called_once_with("/test/path/vector_store")
        mock_print.assert_called_with("Vector store created and saved at /test/path/vector_store.")
    
    @patch('services.FAISS')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.path.dirname')
    def test_create_directory_when_not_exists(self, mock_dirname, mock_makedirs, mock_exists, mock_faiss):
        """Test directory creation when it doesn't exist"""
        mock_chunks = [MagicMock()]
        mock_embeddings = MagicMock()
        mock_vector_store = MagicMock()
        
        mock_faiss.from_documents.return_value = mock_vector_store
        mock_dirname.return_value = "/test/path"
        mock_exists.return_value = False  # Directory doesn't exist
        
        with patch('builtins.print'):
            create_vector_store(mock_chunks, mock_embeddings, "/test/path/vector_store")
        
        mock_makedirs.assert_called_once_with("/test/path")
    
    def test_empty_chunks_list(self):
        """Test behavior with empty chunks list"""
        mock_embeddings = MagicMock()
        
        with patch('builtins.print') as mock_print:
            result = create_vector_store([], mock_embeddings, "/test/path")
        
        assert result is None
        mock_print.assert_called_with("No chunks provided to create vector store.")
    
    def test_no_embeddings_provided(self):
        """Test behavior when no embeddings are provided"""
        mock_chunks = [MagicMock()]
        
        with patch('builtins.print') as mock_print:
            result = create_vector_store(mock_chunks, None, "/test/path")
        
        assert result is None
        mock_print.assert_called_with("Embeddings model is not provided.")
    
    @patch('services.FAISS')
    def test_exception_during_creation(self, mock_faiss):
        """Test exception handling during vector store creation"""
        mock_chunks = [MagicMock()]
        mock_embeddings = MagicMock()
        
        mock_faiss.from_documents.side_effect = Exception("FAISS creation failed")
        
        with patch('builtins.print') as mock_print:
            result = create_vector_store(mock_chunks, mock_embeddings, "/test/path")
        
        assert result is None
        mock_print.assert_called_with("Error creating vector store: FAISS creation failed")


class TestLoadVectorStore:
    """Test cases for load_vector_store function"""
    
    @patch('services.FAISS')
    @patch('os.path.exists')
    def test_successful_vector_store_loading(self, mock_exists, mock_faiss):
        """Test successful vector store loading"""
        mock_exists.return_value = True
        mock_embeddings = MagicMock()
        mock_vector_store = MagicMock()
        
        mock_faiss.load_local.return_value = mock_vector_store
        
        with patch('builtins.print') as mock_print:
            result = load_vector_store("/test/path", mock_embeddings)
        
        assert result == mock_vector_store
        mock_faiss.load_local.assert_called_once_with(
            folder_path="/test/path",
            embeddings=mock_embeddings,
            allow_dangerous_deserialization=True
        )
        mock_print.assert_called_with("Vector store loaded from /test/path.")
    
    @patch('os.path.exists')
    def test_nonexistent_vector_store_path(self, mock_exists):
        """Test behavior when vector store path doesn't exist"""
        mock_exists.return_value = False
        mock_embeddings = MagicMock()
        
        with patch('builtins.print') as mock_print:
            result = load_vector_store("/nonexistent/path", mock_embeddings)
        
        assert result is None
        mock_print.assert_called_with("Vector store path /nonexistent/path does not exist.")
    
    @patch('services.FAISS')
    @patch('os.path.exists')
    def test_exception_during_loading(self, mock_exists, mock_faiss):
        """Test exception handling during vector store loading"""
        mock_exists.return_value = True
        mock_embeddings = MagicMock()
        
        mock_faiss.load_local.side_effect = Exception("Loading failed")
        
        with patch('builtins.print') as mock_print:
            result = load_vector_store("/test/path", mock_embeddings)
        
        assert result is None
        mock_print.assert_called_with("Error loading vector store: Loading failed")


class TestSimilaritySearch:
    """Test cases for similarity_search function"""
    
    def test_successful_similarity_search(self):
        """Test successful similarity search"""
        mock_vector_store = MagicMock()
        mock_results = [MagicMock(), MagicMock(), MagicMock()]
        mock_vector_store.similarity_search.return_value = mock_results
        
        with patch('builtins.print') as mock_print:
            result = similarity_search(mock_vector_store, "test query", k=3)
        
        assert result == mock_results
        mock_vector_store.similarity_search.assert_called_once_with(query="test query", k=3)
        mock_print.assert_called_with("Found 3 similar documents for query: 'test query'.")
    
    def test_similarity_search_with_default_k(self):
        """Test similarity search with default k value"""
        mock_vector_store = MagicMock()
        mock_results = [MagicMock(), MagicMock(), MagicMock()]
        mock_vector_store.similarity_search.return_value = mock_results
        
        with patch('builtins.print'):
            result = similarity_search(mock_vector_store, "test query")
        
        mock_vector_store.similarity_search.assert_called_once_with(query="test query", k=3)
    
    def test_no_vector_store_provided(self):
        """Test behavior when no vector store is provided"""
        with patch('builtins.print') as mock_print:
            result = similarity_search(None, "test query", k=3)
        
        assert result == []
        mock_print.assert_called_with("Vector store is not provided.")
    
    def test_exception_during_search(self):
        """Test exception handling during similarity search"""
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.side_effect = Exception("Search failed")
        
        with patch('builtins.print') as mock_print:
            result = similarity_search(mock_vector_store, "test query", k=3)
        
        assert result == []
        mock_print.assert_called_with("Error during similarity search: Search failed")
    
    def test_search_with_different_k_values(self):
        """Test similarity search with different k values"""
        mock_vector_store = MagicMock()
        mock_results = [MagicMock() for _ in range(5)]
        mock_vector_store.similarity_search.return_value = mock_results
        
        with patch('builtins.print') as mock_print:
            result = similarity_search(mock_vector_store, "test query", k=5)
        
        assert result == mock_results
        mock_vector_store.similarity_search.assert_called_once_with(query="test query", k=5)
        mock_print.assert_called_with("Found 5 similar documents for query: 'test query'.")


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def setup_method(self):
        """Set up temporary directory for each test"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary directory after each test"""
        shutil.rmtree(self.temp_dir)
    
    @patch('services.HuggingFaceEmbeddings')
    @patch('services.FAISS')
    def test_full_workflow(self, mock_faiss, mock_embeddings):
        """Test the complete workflow from embeddings to search"""
        # Mock embeddings
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        
        # Mock vector store
        mock_vector_store = MagicMock()
        mock_faiss.from_documents.return_value = mock_vector_store
        mock_faiss.load_local.return_value = mock_vector_store
        
        # Mock search results
        mock_results = [MagicMock(), MagicMock()]
        mock_vector_store.similarity_search.return_value = mock_results
        
        # Create sample chunks
        chunks = [
            Document(page_content="Sample document 1", metadata={"source": "test1.pdf"}),
            Document(page_content="Sample document 2", metadata={"source": "test2.pdf"})
        ]
        
        vector_store_path = os.path.join(self.temp_dir, "vector_store")
        
        with patch('builtins.print'):
            # Step 1: Get embeddings
            embeddings = get_embeddings("test-model")
            assert embeddings == mock_embedding_instance
            
            # Step 2: Create vector store
            vector_store = create_vector_store(chunks, embeddings, vector_store_path)
            assert vector_store == mock_vector_store
            
            # Step 3: Load vector store
            loaded_store = load_vector_store(vector_store_path, embeddings)
            assert loaded_store == mock_vector_store
            
            # Step 4: Perform similarity search
            results = similarity_search(loaded_store, "test query")
            assert results == mock_results


# Fixtures for reusable test data
@pytest.fixture
def sample_chunks():
    """Create sample document chunks for testing"""
    return [
        Document(
            page_content="This is the first sample document content.",
            metadata={"source": "doc1.pdf", "page": 1}
        ),
        Document(
            page_content="This is the second sample document content.",
            metadata={"source": "doc2.pdf", "page": 1}
        ),
        Document(
            page_content="This is the third sample document content.",
            metadata={"source": "doc3.pdf", "page": 1}
        )
    ]


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings object"""
    return MagicMock()


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store object"""
    return MagicMock()


# Parametrized tests for different model names
@pytest.mark.parametrize("model_name,device", [
    ("sentence-transformers/all-MiniLM-L6-v2", "cpu"),
    ("sentence-transformers/all-mpnet-base-v2", "cpu"),
    ("sentence-transformers/all-distilroberta-v1", "cuda"),
    ("huggingface/CodeBERTa-small-v1", "cpu")
])
def test_various_embedding_models(model_name, device):
    """Test embedding creation with various model names and devices"""
    with patch('services.HuggingFaceEmbeddings') as mock_embeddings:
        mock_embedding_instance = MagicMock()
        mock_embeddings.return_value = mock_embedding_instance
        
        with patch('builtins.print') as mock_print:
            result = get_embeddings(model_name, device)
        
        assert result == mock_embedding_instance
        mock_embeddings.assert_called_once_with(model_name=model_name)
        mock_print.assert_called_with(f"Successfully loaded embeddings model: {model_name} on {device}.")


# Parametrized tests for different k values in similarity search
@pytest.mark.parametrize("k_value,expected_results", [
    (1, 1),
    (3, 3),
    (5, 5),
    (10, 10)
])
def test_similarity_search_k_values(k_value, expected_results):
    """Test similarity search with different k values"""
    mock_vector_store = MagicMock()
    mock_results = [MagicMock() for _ in range(expected_results)]
    mock_vector_store.similarity_search.return_value = mock_results
    
    with patch('builtins.print') as mock_print:
        result = similarity_search(mock_vector_store, "test query", k=k_value)
    
    assert len(result) == expected_results
    mock_vector_store.similarity_search.assert_called_once_with(query="test query", k=k_value)
    mock_print.assert_called_with(f"Found {expected_results} similar documents for query: 'test query'.")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
