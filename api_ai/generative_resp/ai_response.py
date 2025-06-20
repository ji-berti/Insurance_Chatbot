import os
import shutil
from typing import List, Dict

from generative_resp import config_vectordb as config
from generative_resp import config_model as config_model
from generative_resp.pdf_process_utils import load_split_pdfs, load_single_pdf
from generative_resp.services import get_embeddings, create_vector_store, load_vector_store

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain import hub

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# LOAD BASE VECTOR STORE FROM DISK
def load_base_vector_store(force_recreate=False):
    """
    Load or create the base vector store from disk.
    """
    if force_recreate and os.path.exists(config.VECTOR_STORE_PATH):
        shutil.rmtree(config.VECTOR_STORE_PATH)

    if not os.path.exists(config.VECTOR_STORE_PATH):
        print("Creating base vector store on disk...")
        chunks = load_split_pdfs(config.PDF_DIR_POLIZAS, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        embeddings = get_embeddings(config.EMBEDDING_MODEL)
        vector_store = create_vector_store(chunks, embeddings, config.VECTOR_STORE_PATH)
        return vector_store
    else:
        print("Loading existing base vector store from disk...")
        embeddings = get_embeddings(model_name=config.EMBEDDING_MODEL)
        return load_vector_store(config.VECTOR_STORE_PATH, embeddings)

# UPDATE IN MEMORY VECTOR STORE
def update_in_memory_vector_store(file_path: str, vector_store):
    """
    Process a new PDF file and update the in-memory vector store.
    """
    print(f"Updating in-memory vector store with file: {file_path}")
    chunks = load_single_pdf(file_path, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    if chunks:
        vector_store.add_documents(chunks)  # Updates in memory DB
        print("In-memory vector store updated successfully.")
    else:
        print("No documents were added.")

# Generate response considering past history as a parameter
def send_response(query: str, vector_store, conversation_history: List[Dict[str, str]] = None):
    """
    Generate response using conversation history
    
    Args:
        query: User's question
        conversation_history: List of previous messages in the format:
            [{"role": "human", "content": "question"}, {"role": "ai", "content": "answer"}, ...]
    """
    # Create the Gemini LLM instance
    llm = ChatGoogleGenerativeAI(
        model=config_model.GEMINI_MODEL,
        google_api_key=config_model.GEMINI_API_KEY,
        temperature=config_model.TEMPERATURE,
        max_output_tokens=config_model.MAX_OUT_TOKENS
    )

    # Create temporary memory only for the current session
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # If there is previous history, add it to the temporary memory
    if conversation_history:
        for message in conversation_history:
            if message["role"] == "human":
                memory.chat_memory.add_user_message(message["content"])
            elif message["role"] == "ai":
                memory.chat_memory.add_ai_message(message["content"])

    # Retriever tool to search on the vector store
    def retrieve_docs_with_metadata(query: str) -> str:
        """
        Retrieves relevant document chunks and formats them with their metadata.
        """
        docs = vector_store.as_retriever(search_kwargs={'k': config_model.TOP_K}).get_relevant_documents(query)
        
        formatted_results = []
        for doc in docs:
            # Extract document metadata
            source = doc.metadata.get('source', 'Desconocido')
            page = doc.metadata.get('page', 'N/A')
            if isinstance(page, int):
                page += 1

            # Format the result for better understanding of the LLM
            result_str = (
                f"Contenido: {doc.page_content}\n"
                f"Fuente: {os.path.basename(source)}\n"
                f"Página: {page}\n"
                "----------------"
            )
            formatted_results.append(result_str)
        
        return "\n".join(formatted_results)

    retriever_tool = Tool(
        name="VectorDBTool",
        func=retrieve_docs_with_metadata, 
        description=(
            "IMPRESCINDIBLE para buscar información específica en documentos de pólizas de seguros cargados por el usuario. "
            "DEBES usar esta herramienta PRIMERO para cualquier pregunta que no sea un saludo."
        )
    )

    # Search tool to search on the web if the vector store does not contain the answer
    # Function to retrive the source of the information
    def web_search_with_sources(query: str, num_results: int = 3) -> str:
        """
        Makes a web search ans returns the results formated including title, snippet and URL.
        """
        wrapper = DuckDuckGoSearchAPIWrapper()
        results = wrapper.results(query, max_results=num_results)
        
        if not results:
            return "No se encontraron resultados en la web."
            
        formatted_results = []
        for i, res in enumerate(results, 1):
            result_str = (
                f"Resultado {i}:\n"
                f"Título: {res['title']}\n"
                f"Resumen: {res['snippet']}\n"
                f"URL: {res['link']}\n"
                "----------------"
            )
            formatted_results.append(result_str)
        
        return "\n".join(formatted_results)

    search_tool = Tool(
        name="WebSearch",
        func=web_search_with_sources,
        description=(
            "Útil para responder preguntas de conocimiento general. "
            "Usa esta herramienta SOLO si VectorDBTool no encontró una respuesta adecuada."
        )
    )

    tools = [retriever_tool, search_tool]
    
    # BASE PROMPT
    prompt = PromptTemplate(
        template=
        """
        Eres un asistente de IA conversacional y servicial. Tu objetivo es ser educado, informativo y eficiente.
        Considera siempre el historial de la conversación para dar respuestas contextuales.

        **PLAN DE ACCIÓN OBLIGATORIO:**
        1.  **Analiza la pregunta del Humano.**
        2.  **Prioriza `VectorDBTool`:** Para CUALQUIER pregunta que no sea un saludo, DEBES usar `VectorDBTool` primero.
        3.  **Analiza en Profundidad:** Revisa **TODOS** los fragmentos de la `Observation` de `VectorDBTool`. La respuesta completa puede estar distribuida en varios de ellos. Sintetiza la información de todos los fragmentos relevantes para construir la respuesta más completa posible.
        4.  **Usa `WebSearch` como plan B:** Si después de analizar todos los fragmentos de la base de datos, la respuesta sigue siendo insuficiente, y SOLO EN ESE CASO, puedes usar `WebSearch`.
        5.  **Formula la Respuesta Final:** Construye tu respuesta final basándote en tu análisis.

        **FORMATO OBLIGATORIO:**

        Para usar una herramienta, usa este formato:
        ```
        Thought: [Tu razonamiento sobre qué herramienta usar y por qué, siguiendo el PLAN DE ACCIÓN.]
        Action: [El nombre de la herramienta, una de [{tool_names}]]
        Action Input: [El texto de entrada para la herramienta]
        Observation: [El resultado que la herramienta te devuelve]
        ```

        Cuando tengas la respuesta final, o si es un saludo, DEBES usar el siguiente formato. **NO intentes usar herramientas de nuevo.**
        ```
        Thought: [Tu razonamiento final para concluir la respuesta.]
        Final Answer: [Tu respuesta final, completa y bien formada para el usuario.]
        > **Fuentes:**
        > - **Herramienta**: [VectorDBTool, WebSearch o Ninguna]
        > - **Documento**: [Nombre del archivo, si usaste VectorDBTool]
        > - **Página**: [Número de página, si usaste VectorDBTool]
        > - **Enlaces Web**:
        >   - [Si usaste WebSearch, lista cada enlace en una nueva línea con formato `- [Título](URL)`]
        ```

        **COMIENZA AHORA**

        Historial de la conversación anterior:
        {chat_history}

        Nueva entrada del Humano: {input}
        Thought:{agent_scratchpad}
        """,
        input_variables=["input", "agent_scratchpad", "chat_history"],
        partial_variables={
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
            "tool_names": ", ".join([tool.name for tool in tools])
        }
    )

    # Create the agent using the LLM, tools, and prompt
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory, 
        verbose=True,
        handle_parsing_errors="Por favor, reformula tu respuesta en el formato correcto.",
        # max_iterations=8, # Avoid too many iterations
        # early_stopping_method="generate"
    )

    # Execute the request
    result = agent_executor.invoke({"input": query})
    return result["output"]