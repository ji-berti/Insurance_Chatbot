import os

PDF_DIR_POLIZAS = 'data/polizas'

VECTOR_STORE_OUTPUT = 'data/vector_store'
if not os.path.exists(VECTOR_STORE_OUTPUT):
    os.makedirs(VECTOR_STORE_OUTPUT)

VECTOR_STORE_NAME = 'faiss_index_polizas'

VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_OUTPUT, VECTOR_STORE_NAME)

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
