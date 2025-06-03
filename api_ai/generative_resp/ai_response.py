import os
import shutil

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

# Create or load the vector store index
def create_index(force_recreate=False):
    # Recreate the vector store if specified
    if force_recreate and os.path.exists(config.VECTOR_STORE_PATH):
        shutil.rmtree(config.VECTOR_STORE_PATH)

    # If the vector store does not exist, create it
    if not os.path.exists(config.VECTOR_STORE_PATH):
        print("Creating vector store index...")
        chunks = load_split_pdfs( # Load the chunks
            pdf_dir=config.PDF_DIR_POLIZAS,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        embeddings = get_embeddings(model_name=config.EMBEDDING_MODEL) # Get the embeddings
        vector_store = create_vector_store( # Create the vector store
            chunks=chunks,
            embeddings=embeddings,
            vector_store_path=config.VECTOR_STORE_PATH
        )
        return vector_store
    
    # If the vector store already exists, load it
    else:
        print("Loading existing vector store index...")
        embeddings = get_embeddings(model_name=config.EMBEDDING_MODEL)
        return load_vector_store(vector_store_path=config.VECTOR_STORE_PATH, embeddings=embeddings)

# Add a new PDF file to the existing vector store
def update_vector_store_with_new_file(file_path: str, vector_store):
    if not file_path.endswith(".pdf") or not os.path.exists(file_path):
        raise ValueError(f"{file_path} is not a valid PDF file.")

    print(f"Processing new file to add to the vector store: {file_path}")
    chunks = load_single_pdf( # Load the chunks from the new PDF file
        pdf_path=file_path,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    if chunks:
        vector_store.add_documents(chunks)
        print("File added successfully to the vector store.")
    else:
        print("No documents were added to the vector store.")

# Generate the agent with the vector store and tools
def generate_agent(vector_store):
    # Create the Gemini LLM instance
    llm = ChatGoogleGenerativeAI(
        model=config_model.GEMINI_MODEL,
        google_api_key=config_model.GEMINI_API_KEY,
        temperature=config_model.TEMPERATURE,
        max_output_tokens=config_model.MAX__OUT_TOKENS
    )

    # Create the memory cache for the conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Retriever tool to search on the vector store
    retriever_tool = Tool(
        name="VectorDBTool",
        func=lambda q: "\n".join(
            [doc.page_content for doc in vector_store.as_retriever().get_relevant_documents(q)]
        ),
        description=(
            "Usa esta herramienta para buscar información en la base de datos de pólizas de seguros. "
            "Use this tool to search the insurance policy database."
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

    # Load the prompt template from the hub or create a default one
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        prompt = PromptTemplate(
            template="""Answer the following questions as best you can. You have access to the following tools:

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!

                Question: {input}
                Thought:{agent_scratchpad}""",
            input_variables=["input", "agent_scratchpad"],
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

    return agent_executor

# Retrieve a response for a given query using the agent
def send_response(query: str):
    vector_store = create_index(force_recreate=False)
    agent = generate_agent(vector_store)
    result = agent.invoke({"input": query})
    return result["output"]
