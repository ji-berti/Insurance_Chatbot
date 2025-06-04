# Insurance_Chatbot


# To run the test install dependencies
pip install pytest pytest-mock
# Run the tests for the pdf_process_utils.py
# Run all tests with verbose output
pytest test_pdf_process_utils.py -v
# Run specific test class
pytest test_pdf_process_utils.py::TestLoadSplitPdfs -v
# Run with coverage report (if you have pytest-cov installed)
pytest test_pdf_process_utils.py --cov=pdf_process_utils

# Run the tests for the services.py
# Run all tests with verbose output
pytest test_services.py -v
# Run specific test class
pytest test_services.py::TestGetEmbeddings -v
# Run parametrized tests only
pytest test_services.py -k "parametrize" -v
# Run with coverage report (if you have pytest-cov installed)
pytest test_services.py --cov=services --cov-report=html
