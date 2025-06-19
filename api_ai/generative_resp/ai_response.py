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
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain import hub

# LOAD BASE VECTOR STORE FROM DISK
def load_base_vector_store(force_recreate=False):
    """
    Load or create the base vector store from disk.
    """
    if force_recreate and os.path.exists(config.VECTOR_STORE_PATH):
        shutil.rmtree(config.VECTOR_STORE_PATH)

    if not os.path.exists(config.VECTOR_STORE_PATH):
        print("Creating base vector store on disk...")
        chunks = load_split_pdfs(...)
        embeddings = get_embeddings(...)
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
            "Usa esta herramienta para buscar información en la base de datos de pólizas de seguros. "
            "La herramienta devolverá el contenido del documento, el nombre del archivo (Fuente) y el número de página."
        )
    )

    # Search tool to search on the web if the vector store does not contain the answer
    search_tool = Tool(
        name="WebSearch",
        func=DuckDuckGoSearchRun().run,
        description=(
            "Usa esta herramienta para buscar información en Internet si la base de datos no contiene la respuesta. "
            "Use this tool to search the internet if the database lacks the answer."
        )
    )

    tools = [retriever_tool, search_tool]
    
    # BASE PROMPT
    prompt = PromptTemplate(
        template=
        """
        Assistant is a helpful and conversational AI designed to answer questions about insurance policies.
        It has access to a database of insurance policies and can also search the web for general information.
        Strive to be polite and informative. If you don't know the answer from the provided tools, say so.
        
        You should consider the conversation history to provide contextual and coherent responses.
        If the user refers to something mentioned earlier in the conversation, use that context.
        Pay attention to pronouns like "it", "that", "this" that might refer to previous topics.

        TOOLS:
        ------
        You have access to the following tools:
        {tools}

        To use a tool, you MUST use the following format:

        Thought: Do I need to use a tool? Yes
        Action: The action to take, should be one of [{tool_names}]
        Action Input: The input to the action
        Observation: The result of the action

        (This Thought/Action/Action Input/Observation can repeat N times)

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format below.
        You must add a "Fuentes:" section at the end of your response.
        - If you used VectorDBTool, you MUST cite the 'Fuente' and 'Página' from the observation.
        - If you used WebSearch, you MUST state that you used WebSearch.
        - If you did not use any tool, say so.

        Thought: Do I need to use a tool? No
        Final Answer: [Tu respuesta final y completa aquí.]

        > **Fuentes:**
        - **Herramienta**: [Nombre de la herramienta usada: VectorDBTool, WebSearch o Ninguna]
        - **Documento**: [Nombre del archivo (ej. poliza_vida.pdf), si usaste VectorDBTool]
        - **Página**: [Número de página, si usaste VectorDBTool]

        Begin!

        Previous conversation history:
        {chat_history}

        New input from Human: {input}
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
        handle_parsing_errors=True
    )

    # Execute the request
    result = agent_executor.invoke({"input": query})
    return result["output"]