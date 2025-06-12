import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys

# Import the module to test (adjust the import path as needed)
# Assuming your script is named 'ai_response.py'
try:
    import ai_response
except ImportError:
    # If the module is in a different location, adjust this
    sys.path.append('path/to/your/module')
    import ai_response

class TestAIResponse:
    """Test suite for the AI Response system"""
    
    def setup_method(self):
        """Setup method called before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_vector_store_path = os.path.join(self.temp_dir, "test_vector_store")
        self.test_pdf_path = os.path.join(self.temp_dir, "test_policy.pdf")
        
    def teardown_method(self):
        """Cleanup method called after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('ai_response.config.VECTOR_STORE_PATH')
    @patch('ai_response.load_split_pdfs')
    @patch('ai_response.get_embeddings')
    @patch('ai_response.create_vector_store')
    def test_create_index_new_vector_store(self, mock_create_vs, mock_get_emb, mock_load_pdfs, mock_path):
        """Test creating a new vector store when none exists"""
        # Setup mocks
        mock_path.return_value = self.test_vector_store_path
        mock_chunks = [Mock(), Mock()]
        mock_load_pdfs.return_value = mock_chunks
        mock_embeddings = Mock()
        mock_get_emb.return_value = mock_embeddings
        mock_vector_store = Mock()
        mock_create_vs.return_value = mock_vector_store
        
        # Execute
        result = ai_response.create_index(force_recreate=False)
        
        # Verify
        mock_load_pdfs.assert_called_once()
        mock_get_emb.assert_called_once()
        mock_create_vs.assert_called_once_with(
            chunks=mock_chunks,
            embeddings=mock_embeddings,
            vector_store_path=self.test_vector_store_path
        )
        assert result == mock_vector_store
    
    @patch('ai_response.config.VECTOR_STORE_PATH')
    @patch('ai_response.get_embeddings')
    @patch('ai_response.load_vector_store')
    @patch('os.path.exists')
    def test_create_index_existing_vector_store(self, mock_exists, mock_load_vs, mock_get_emb, mock_path):
        """Test loading an existing vector store"""
        # Setup mocks
        mock_path.return_value = self.test_vector_store_path
        mock_exists.return_value = True
        mock_embeddings = Mock()
        mock_get_emb.return_value = mock_embeddings
        mock_vector_store = Mock()
        mock_load_vs.return_value = mock_vector_store
        
        # Execute
        result = ai_response.create_index(force_recreate=False)
        
        # Verify
        mock_load_vs.assert_called_once_with(
            vector_store_path=self.test_vector_store_path,
            embeddings=mock_embeddings
        )
        assert result == mock_vector_store
    
    @patch('ai_response.config.VECTOR_STORE_PATH')
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_create_index_force_recreate(self, mock_rmtree, mock_exists, mock_path):
        """Test force recreating the vector store"""
        # Setup mocks
        mock_path.return_value = self.test_vector_store_path
        mock_exists.return_value = True
        
        with patch.object(ai_response, 'load_split_pdfs'), \
             patch.object(ai_response, 'get_embeddings'), \
             patch.object(ai_response, 'create_vector_store') as mock_create:
            
            mock_create.return_value = Mock()
            
            # Execute
            ai_response.create_index(force_recreate=True)
            
            # Verify
            mock_rmtree.assert_called_once_with(self.test_vector_store_path)
    
    @patch('ai_response.load_single_pdf')
    @patch('os.path.exists')
    def test_update_vector_store_with_new_file_success(self, mock_exists, mock_load_pdf):
        """Test successfully adding a new PDF to vector store"""
        # Setup
        mock_exists.return_value = True
        mock_chunks = [Mock(), Mock()]
        mock_load_pdf.return_value = mock_chunks
        mock_vector_store = Mock()
        
        # Execute
        ai_response.update_vector_store_with_new_file(self.test_pdf_path, mock_vector_store)
        
        # Verify
        mock_load_pdf.assert_called_once()
        mock_vector_store.add_documents.assert_called_once_with(mock_chunks)
    
    def test_update_vector_store_invalid_file(self):
        """Test error handling for invalid file"""
        mock_vector_store = Mock()
        
        # Test non-PDF file
        with pytest.raises(ValueError, match="is not a valid PDF file"):
            ai_response.update_vector_store_with_new_file("test.txt", mock_vector_store)
        
        # Test non-existent file
        with pytest.raises(ValueError, match="is not a valid PDF file"):
            ai_response.update_vector_store_with_new_file("nonexistent.pdf", mock_vector_store)
    
    @patch('ai_response.load_single_pdf')
    @patch('os.path.exists')
    def test_update_vector_store_no_chunks(self, mock_exists, mock_load_pdf):
        """Test when PDF processing returns no chunks"""
        # Setup
        mock_exists.return_value = True
        mock_load_pdf.return_value = []  # No chunks
        mock_vector_store = Mock()
        
        # Execute
        ai_response.update_vector_store_with_new_file(self.test_pdf_path, mock_vector_store)
        
        # Verify
        mock_vector_store.add_documents.assert_not_called()
    
    @patch('ai_response.ChatGoogleGenerativeAI')
    @patch('ai_response.ConversationBufferMemory')
    @patch('ai_response.Tool')
    @patch('ai_response.create_react_agent')
    @patch('ai_response.AgentExecutor')
    def test_generate_agent(self, mock_agent_executor, mock_create_agent, mock_tool, 
                           mock_memory, mock_llm_class):
        """Test agent generation"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_memory_instance = Mock()
        mock_memory.return_value = mock_memory_instance
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_executor = Mock()
        mock_agent_executor.return_value = mock_executor
        
        # Execute
        result = ai_response.generate_agent(mock_vector_store)
        
        # Verify
        mock_llm_class.assert_called_once()
        mock_memory.assert_called_once()
        mock_create_agent.assert_called_once()
        mock_agent_executor.assert_called_once()
        assert result == mock_executor
    
    @patch('ai_response.create_index')
    @patch('ai_response.generate_agent')
    def test_send_response(self, mock_generate_agent, mock_create_index):
        """Test sending a response"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_create_index.return_value = mock_vector_store
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "Test response"}
        mock_generate_agent.return_value = mock_agent
        
        # Execute
        result = ai_response.send_response("Test query")
        
        # Verify
        mock_create_index.assert_called_once_with(force_recreate=False)
        mock_generate_agent.assert_called_once_with(mock_vector_store)
        mock_agent.invoke.assert_called_once_with({"input": "Test query"})
        assert result == "Test response"


class TestIntegration:
    """Integration tests (require actual dependencies)"""
    
    @pytest.mark.integration
    @patch('ai_response.config')
    def test_full_workflow_mock(self, mock_config):
        """Test the full workflow with mocked dependencies"""
        # This test requires more setup and should be run in an environment
        # where all dependencies are available
        pass


# Test utilities and fixtures
@pytest.fixture
def sample_pdf_content():
    """Fixture providing sample PDF content for testing"""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"

@pytest.fixture
def mock_vector_store():
    """Fixture providing a mock vector store"""
    vector_store = Mock()
    vector_store.as_retriever.return_value.get_relevant_documents.return_value = [
        Mock(page_content="Sample document content 1"),
        Mock(page_content="Sample document content 2")
    ]
    return vector_store


# Performance tests
class TestPerformance:
    """Performance-related tests"""
    
    @pytest.mark.performance
    def test_response_time(self):
        """Test that responses are generated within acceptable time limits"""
        import time
        
        with patch('ai_response.create_index'), \
             patch('ai_response.generate_agent') as mock_agent:
            
            mock_agent_instance = Mock()
            mock_agent_instance.invoke.return_value = {"output": "Fast response"}
            mock_agent.return_value = mock_agent_instance
            
            start_time = time.time()
            result = ai_response.send_response("Quick test")
            end_time = time.time()
            
            # Assert response time is under 5 seconds (adjust as needed)
            assert (end_time - start_time) < 5.0
            assert result == "Fast response"


# Error handling tests
class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('ai_response.create_index')
    def test_send_response_with_exception(self, mock_create_index):
        """Test error handling in send_response"""
        mock_create_index.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception, match="Database connection failed"):
            ai_response.send_response("Test query")


if __name__ == "__main__":
    # Run the tests
    print("Running AI Response Tests...")
    
    # Basic test runner (you can also use pytest command line)
    import unittest
    
    # Convert pytest tests to unittest format for basic running
    suite = unittest.TestSuite()
    
    # Add basic functionality tests
    print("\n=== Running Basic Functionality Tests ===")
    try:
        # You can add simple direct function calls here for quick testing
        print("✓ Test framework setup complete")
        print("✓ All imports successful")
        print("\nTo run full test suite, use: pytest test_ai_response.py")
        print("To run specific test categories:")
        print("  pytest test_ai_response.py::TestAIResponse")
        print("  pytest test_ai_response.py -m integration")
        print("  pytest test_ai_response.py -m performance")
        
    except Exception as e:
        print(f"✗ Setup failed: {e}")

    print("\n=== Test Configuration Complete ===")
