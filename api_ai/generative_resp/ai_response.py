import os
import shutil

from generative_resp import config_vectordb as config
from generative_resp import config_model as config_model
from generative_resp.pdf_process_utils import load_split_pdfs
from generative_resp.services import get_embeddings, create_vector_store, load_vector_store

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain import hub


# Create o load the vector store
def create_index(force_recreate=False):
    # If the recreate de VS
    if force_recreate and os.path.exists(config.VECTOR_STORE_PATH):
        shutil.rmtree(config.VECTOR_STORE_PATH)

    # If the vector store does not exist, create it
    if not os.path.exists(config.VECTOR_STORE_PATH):
        print("Creating vector store index...")
        chunks = load_split_pdfs( # Obtain the chunks from the PDF files
            pdf_dir=config.PDF_DIR_POLIZAS,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        # Get the embeddings and create the vector store
        embeddings = get_embeddings(model_name=config.EMBEDDING_MODEL)
        vector_store = create_vector_store(
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

# Generate the agent with the vector store and tools
def generate_agent(vector_store):
    # Call the Google Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=config_model.GEMINI_MODEL,
        google_api_key=config_model.GEMINI_API_KEY,
        temperature=config_model.TEMPERATURE,
        max_output_tokens=config_model.MAX__OUT_TOKENS
    )

    # Memory history for the agent
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Avialable tools 
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

    search_tool = Tool(
        name="WebSearch",
        func=DuckDuckGoSearchRun().run,
        description=(
            "Usa esta herramienta para buscar información en Internet si la base de datos no contiene la respuesta. "
            "Use this tool to search the internet if the database lacks the answer."
        )
    )

    tools = [retriever_tool, search_tool]

    # Create the base prompt for the agent
    # Option 1: Pull the prompt from the langchain hub
    try:
        prompt = hub.pull("hwchase17/react")
    except:
        # Option 2: Create a custom promt
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

    # Crear el agente con el prompt
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Ejecutar agente
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor



def send_response(query: str):
    vector_store = create_index(force_recreate=False)
    agent = generate_agent(vector_store)
    result = agent.invoke({"input": query})
    return result["output"]