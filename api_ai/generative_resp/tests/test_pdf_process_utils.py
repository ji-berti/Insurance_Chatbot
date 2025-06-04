import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil
from langchain.schema import Document

# Import the functions to test
# Assuming your original script is saved as 'pdf_process_utils.py'
from pdf_process_utils import load_split_pdfs, load_single_pdf


class TestLoadSplitPdfs:
    """Test cases for load_split_pdfs function"""
    
    def test_nonexistent_directory(self):
        """Test behavior when directory doesn't exist"""
        with pytest.raises(FileNotFoundError, match="The directory /nonexistent/path does not exist"):
            load_split_pdfs("/nonexistent/path", 1000, 200)
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_empty_directory(self, mock_listdir, mock_exists):
        """Test behavior when directory exists but has no PDF files"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['file1.txt', 'file2.doc']
        
        with patch('builtins.print') as mock_print:
            result = load_split_pdfs("/test/dir", 1000, 200)
            
        assert result == []
        mock_print.assert_called_with("No PDF files found in the directory.")
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('pdf_process_utils.PyPDFLoader')
    @patch('pdf_process_utils.RecursiveCharacterTextSplitter')
    def test_successful_processing(self, mock_splitter, mock_loader, mock_listdir, mock_exists):
        """Test successful processing of PDF files"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ['file1.pdf', 'file2.pdf', 'file3.txt']
        
        # Mock document objects
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "This is content from document 1"
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "This is content from document 2"
        
        # Mock loader instances
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]
        mock_loader.return_value = mock_loader_instance
        
        # Mock text splitter
        mock_splitter_instance = MagicMock()
        mock_chunks = [MagicMock(), MagicMock(), MagicMock()]
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance
        
        with patch('builtins.print') as mock_print:
            result = load_split_pdfs("/test/dir", 1000, 200)
        
        # Assertions
        assert result == mock_chunks
        assert mock_loader.call_count == 2  # Only 2 PDF files
        mock_splitter.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        mock_print.assert_called_with("Split 4 documents into 3 chunks.")
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('pdf_process_utils.PyPDFLoader')
    def test_loader_exception_handling(self, mock_loader, mock_listdir, mock_exists):
        """Test handling of loader exceptions"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['file1.pdf', 'file2.pdf']
        
        # First loader raises exception, second succeeds
        mock_loader_instance1 = MagicMock()
        mock_loader_instance1.load.side_effect = Exception("PDF loading error")
        
        mock_doc = MagicMock()
        mock_doc.page_content = "Valid content"
        mock_loader_instance2 = MagicMock()
        mock_loader_instance2.load.return_value = [mock_doc]
        
        mock_loader.side_effect = [mock_loader_instance1, mock_loader_instance2]
        
        with patch('pdf_process_utils.RecursiveCharacterTextSplitter') as mock_splitter:
            mock_splitter_instance = MagicMock()
            mock_chunks = [MagicMock()]
            mock_splitter_instance.split_documents.return_value = mock_chunks
            mock_splitter.return_value = mock_splitter_instance
            
            with patch('builtins.print') as mock_print:
                result = load_split_pdfs("/test/dir", 1000, 200)
        
        # Should continue processing despite first file error
        assert result == mock_chunks
        mock_print.assert_any_call("Error loading /test/dir/file1.pdf: PDF loading error")
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('pdf_process_utils.PyPDFLoader')
    def test_empty_documents(self, mock_loader, mock_listdir, mock_exists):
        """Test behavior when documents have no valid content"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['file1.pdf']
        
        # Mock documents with empty/invalid content
        mock_doc1 = MagicMock()
        mock_doc1.page_content = ""  # Empty content
        mock_doc2 = MagicMock()
        mock_doc2.page_content = "   "  # Whitespace only
        mock_doc3 = MagicMock()
        mock_doc3.page_content = None  # None content
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc1, mock_doc2, mock_doc3]
        mock_loader.return_value = mock_loader_instance
        
        with patch('builtins.print') as mock_print:
            result = load_split_pdfs("/test/dir", 1000, 200)
        
        assert result == []
        mock_print.assert_called_with("No valid documents found.")


class TestLoadSinglePdf:
    """Test cases for load_single_pdf function"""
    
    def test_nonexistent_file(self):
        """Test behavior when file doesn't exist"""
        with pytest.raises(FileNotFoundError, match="The file /nonexistent/file.pdf does not exist"):
            load_single_pdf("/nonexistent/file.pdf", 1000, 200)
    
    @patch('os.path.exists')
    @patch('pdf_process_utils.PyPDFLoader')
    @patch('pdf_process_utils.RecursiveCharacterTextSplitter')
    def test_successful_processing(self, mock_splitter, mock_loader, mock_exists):
        """Test successful processing of a single PDF file"""
        mock_exists.return_value = True
        
        # Mock document
        mock_doc = MagicMock()
        mock_doc.page_content = "This is valid PDF content"
        
        # Mock loader
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance
        
        # Mock text splitter
        mock_splitter_instance = MagicMock()
        mock_chunks = [MagicMock(), MagicMock()]
        mock_splitter_instance.split_documents.return_value = mock_chunks
        mock_splitter.return_value = mock_splitter_instance
        
        with patch('builtins.print') as mock_print:
            result = load_single_pdf("/test/file.pdf", 1000, 200)
        
        assert result == mock_chunks
        mock_loader.assert_called_once_with("/test/file.pdf")
        mock_splitter.assert_called_once_with(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        mock_print.assert_called_with("Created 2 chunks from file /test/file.pdf.")
    
    @patch('os.path.exists')
    @patch('pdf_process_utils.PyPDFLoader')
    def test_loader_exception(self, mock_loader, mock_exists):
        """Test handling of loader exceptions"""
        mock_exists.return_value = True
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.side_effect = Exception("PDF loading failed")
        mock_loader.return_value = mock_loader_instance
        
        with patch('builtins.print') as mock_print:
            result = load_single_pdf("/test/file.pdf", 1000, 200)
        
        assert result == []
        mock_print.assert_called_with("Error loading PDF /test/file.pdf: PDF loading failed")
    
    @patch('os.path.exists')
    @patch('pdf_process_utils.PyPDFLoader')
    def test_no_valid_content(self, mock_loader, mock_exists):
        """Test behavior when PDF has no valid content"""
        mock_exists.return_value = True
        
        # Mock documents with invalid content
        mock_doc1 = MagicMock()
        mock_doc1.page_content = ""
        mock_doc2 = MagicMock()
        mock_doc2.page_content = None
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]
        mock_loader.return_value = mock_loader_instance
        
        with patch('builtins.print') as mock_print:
            result = load_single_pdf("/test/file.pdf", 1000, 200)
        
        assert result == []
        mock_print.assert_called_with("No valid content in /test/file.pdf.")


class TestIntegration:
    """Integration tests with real file system operations"""
    
    def setup_method(self):
        """Set up temporary directory for each test"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary directory after each test"""
        shutil.rmtree(self.temp_dir)
    
    def test_directory_operations(self):
        """Test actual directory operations"""
        # Test with empty directory
        with patch('pdf_process_utils.PyPDFLoader'):
            result = load_split_pdfs(self.temp_dir, 1000, 200)
            assert result == []
        
        # Create some non-PDF files
        with open(os.path.join(self.temp_dir, 'test.txt'), 'w') as f:
            f.write("Not a PDF")
        
        with patch('pdf_process_utils.PyPDFLoader'):
            result = load_split_pdfs(self.temp_dir, 1000, 200)
            assert result == []


# Fixtures for reusable test data
@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    return Document(
        page_content="This is a sample document content for testing purposes.",
        metadata={"source": "test.pdf", "page": 1}
    )


@pytest.fixture
def sample_documents():
    """Create multiple sample documents for testing"""
    return [
        Document(
            page_content="First document content",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        Document(
            page_content="Second document content",
            metadata={"source": "test2.pdf", "page": 1}
        )
    ]


# Parametrized tests for different chunk sizes
@pytest.mark.parametrize("chunk_size,chunk_overlap", [
    (100, 20),
    (500, 50),
    (1000, 100),
    (2000, 200)
])
def test_chunk_parameters(chunk_size, chunk_overlap):
    """Test that different chunk parameters are passed correctly"""
    with patch('os.path.exists') as mock_exists, \
         patch('pdf_process_utils.PyPDFLoader') as mock_loader, \
         patch('pdf_process_utils.RecursiveCharacterTextSplitter') as mock_splitter:
        
        mock_exists.return_value = True
        
        # Mock document
        mock_doc = MagicMock()
        mock_doc.page_content = "Valid content"
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance
        
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = [MagicMock()]
        mock_splitter.return_value = mock_splitter_instance
        
        load_single_pdf("/test/file.pdf", chunk_size, chunk_overlap)
        
        mock_splitter.assert_called_once_with(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
