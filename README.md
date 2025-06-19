# Insurance Policy AI Assistant

![Project Badge](https://img.shields.io/badge/Project-AI%20Chatbot-blue)
![Python Badge](https://img.shields.io/badge/Python-3.10-blue)
![Docker Badge](https://img.shields.io/badge/Docker-Enabled-blue)
![LangChain Badge](https://img.shields.io/badge/Framework-LangChain-purple)
![Gemini Badge](https://img.shields.io/badge/LLM-Gemini-purple)
![Streamlit Badge](https://img.shields.io/badge/UI-Streamlit-orange)

An intelligent, multi-tool agent designed to revolutionize the insurance industry by optimizing policy creation, risk assessment, and information retrieval.

---

## Table of Contents
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Project Team](#-project-team)

---

## 🎯 The Problem

In the insurance and finance industries, the underwriting process—evaluating and analyzing risks to approve or reject a policy—is fundamental. This process, along with the generation of new policy documents, is often time-consuming, repetitive, and requires meticulous analysis of existing policies and market data. General policies, which cover a wide range of common risks, are standardized, but adapting them or creating new ones requires efficient access to a vast knowledge base. This manual effort can lead to slower customer response times and increased operational costs.

## 💡 Our Solution

This project introduces an **AI-powered Chatbot** that acts as an intelligent assistant and advisor for insurance companies. By leveraging Large Language Models (LLMs) and a multi-tool agent architecture, our chatbot streamlines the entire policy creation and inquiry lifecycle. It can instantly access a database of previous policies to answer complex questions, generate new policy clauses, and even search the web for the latest industry news, providing concise summaries to keep agents informed.

This tool transforms a tedious process into a dynamic, conversational experience, ensuring that insurance professionals have reliable, accurate information at their fingertips.

## ✨ Key Features

- **Conversational Q&A:** Ask complex questions about existing insurance policies in natural language.
- **Dynamic Knowledge Base (RAG):** The chatbot uses Retrieval-Augmented Generation (RAG) to pull information from a vector database of insurance policies.
- **Session-Based Document Upload:** Users can upload a PDF policy for the duration of a session. The agent will prioritize this document for its answers, enabling analysis of policies not yet in the main database.
- **Live Web Search:** The agent can access the internet to find the latest news, regulations, or information related to insurance companies and policies.
- **Source Citing:** The agent always cites its sources, whether it's a specific page in a document from the vector database or a web search result.
- **Stateful Conversations:** The chatbot remembers previous interactions within a session to provide contextual and coherent responses.
- **Dockerized & Scalable:** The entire application is containerized with Docker, ensuring easy deployment and scalability.

---

## 🏗️ System Architecture

The project is built on a decoupled, two-service architecture (frontend and backend) orchestrated by Docker Compose. This ensures modularity and scalability.

**1. Frontend (Streamlit):**
   - A user-friendly web interface built with Streamlit.
   - Manages user sessions and conversation history.
   - Allows users to upload PDF documents for temporary, session-based analysis.
   - Communicates with the backend via HTTP requests.

**2. Backend (FastAPI):**
   - A robust API built with FastAPI that exposes endpoints for the chatbot's logic.
   - It hosts the LangChain Agent, which is the core of the system.
   - Manages a base vector store (FAISS) loaded at startup and creates separate, in-memory vector stores for each user session to handle uploaded files.

**Data Flow:**

1.  A user sends a message or uploads a file through the **Streamlit UI**.
2.  The UI sends a request containing the query, session ID, and conversation history to the `/ask` or `/upload` endpoint on the **FastAPI Backend**.
3.  The backend identifies the user's session and directs the request to the corresponding **LangChain Agent**.
4.  The Agent, powered by **Google's Gemini model**, processes the query. Based on the query, it decides which tool to use:
    - **VectorDBTool:** Searches the FAISS vector store (base + session-specific) for relevant policy information.
    - **WebSearch Tool:** Uses DuckDuckGo to search the internet for real-time information.
5.  The chosen tool returns its findings to the agent.
6.  The agent synthesizes the information and formulates a final answer, including the source.
7.  The FastAPI backend sends the final answer back to the Streamlit UI, which displays it to the user.

![Architecture Diagram](https://i.imgur.com/your-architecture-diagram.png) 
*Note: Placeholder for a real architecture diagram.*

---

## 🛠️ Tech Stack

| Category              | Technology / Tool                                     |
| --------------------- | ----------------------------------------------------- |
| **AI / NLP Framework**| LangChain                                             |
| **LLM Provider** | Google Gemini API                                     |
| **Embeddings Model** | Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)|
| **Vector Database** | FAISS (Facebook AI Similarity Search)                 |
| **Backend** | FastAPI                                               |
| **Frontend** | Streamlit                                             |
| **Web Search Tool** | DuckDuckGo                                            |
| **Containerization** | Docker, Docker Compose                                |
| **Language** | Python 3.10                                           |

---

## 📁 Project Structure

The repository is organized into two main components: the backend API and the frontend application.

```
├── api_ai/
│   ├── generative_resp/
│   │   ├── ai_response.py       # Core agent logic, prompts, and tool definitions.
│   │   ├── config_model.py      # Configuration for the Gemini model.
│   │   ├── config_vectordb.py   # Configuration for the vector database.
│   │   ├── pdf_process_utils.py # Utilities for loading and splitting PDFs.
│   │   ├── services.py          # Services for embeddings and vector store operations.
│   │   └── policies/            # Default PDF policies to build the base vector store.
│   ├── api.py                   # FastAPI application, endpoints, and session management.
│   └── Dockerfile               # Dockerfile for the backend service.
│   └── .env                         # For storing secret keys (e.g., Gemini API Key).
│
├── ui/
│   ├── app.py                   # The Streamlit frontend application code.
│   └── Dockerfile               # Dockerfile for the frontend service.
│
├── .gitignore
├── docker-compose.yml           # Orchestrates the startup of all services.
└── EDA.ipynb                    # Base dataset EDA
└── README.md                    # You are here!
```

---

## 🚀 Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

You must have the following installed:
- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/ji-berti/Insurance_Chatbot.git
    
    cd Insurance_Chatbot/
    ```

2.  **Create the environment file:**
    Create a file named `.env` in the `/api_ai` folder of the project directory. This file will hold your Google Gemini API key.
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
    You can obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

3.  **Build and run the application:**
    The entire application stack (backend and frontend) is orchestrated by Docker Compose. Simply run the following command from the root directory:
    ```sh
    docker compose up --build
    ```
    - The `--build` flag ensures that the Docker images are rebuilt if there are any changes to the code or dependencies.
    - The first time you run this, it will take some time to download the base Docker images and the embedding models. The script will also automatically create the initial vector database from the PDFs located in `api_ai/generative_resp/policies/`.

4.  **Access the application:**
    Once the containers are up and running, you can access the chatbot's web interface by navigating to:
    - **`http://localhost:8501`**

That's it! You can now start interacting with your AI Insurance Assistant.

---

## 👥 Project Team

This project was developed by:

- **Luis Mora** - [ljmor](https://github.com/ljmor)
- **Juan Ignacio Berti** - [ji-berti](https://github.com/ji-berti)
- **Nicole Condori** - [Nicolecondori2](https://github.com/Nicolecondori2)
- **Harold Alvarez Restrepo** - [haroldalre](https://github.com/haroldalre)
- **Jose Samuel Alvarez Silva**