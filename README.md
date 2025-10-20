# AI Pipeline with LangGraph, LangChain, Qdrant and Streamlit

This project demonstrates how to build a simple agentic AI pipeline using **LangChain**, **LangGraph**, **LangSmith**, **Qdrant** and **Streamlit**.  It guides you through the complete lifecycle of a Retrieval‑Augmented Generation (RAG) application that can answer questions from a PDF document and also fetch real‑time weather data via the OpenWeatherMap API.

## Features

1. **Agentic decision making** – The pipeline uses LangGraph to decide whether a user’s question is about the weather or about the contents of a PDF.  LangGraph acts as a conductor that directs the flow between multiple components and decides when to retrieve information and when to generate responses.  Retrieval agents are useful when you want an LLM to decide whether to fetch context from a vector store or respond directly.

2. **Retrieval‑Augmented Generation (RAG)** – The system implements the standard RAG workflow: split a document into manageable chunks, store their embeddings in a vector database (Qdrant), retrieve the most relevant chunks for a query and use them to answer the question.  Qdrant provides efficient similarity search over dense vector embeddings, and LangChain integrates seamlessly with Qdrant to create collections and perform similarity searches.

3. **Weather retrieval** – For weather‑related questions the pipeline calls the OpenWeatherMap API.  The API returns current weather data in JSON format and requires latitude, longitude and an API key.  A simple extraction function tries to identify the city from the user question and fetches the current conditions.

4. **Embeddings and vector storage** – Embeddings are generated via `OpenAIEmbeddings` (you can switch to other providers) and stored in a local Qdrant collection.  The code demonstrates how to create an in‑memory or on‑disk collection, add documents and query them.

5. **LangSmith evaluation** – LangSmith can be used to trace and evaluate the LLM’s responses.  The `LangSmith` SDK’s `evaluate()` function allows you to score outputs against a labelled dataset.  This repository includes example scaffolding to run evaluations and capture traces.

6. **Streamlit UI** – A simple Streamlit front‑end demonstrates the application.  The UI captures user questions, displays chat history and streams responses.  The `st.chat_input()` and `st.chat_message()` functions make it easy to build chat applications.

## Folder structure

```text
ai_pipeline_project/
├── README.md            # this file
├── requirements.txt     # Python dependencies
├── .env.example         # example environment variables
├── data/
│   └── sample.pdf       # example PDF (generated automatically if missing)
├── src/
│   ├── __init__.py
│   ├── graph.py         # builds the LangGraph pipeline
│   ├── rag.py           # helper functions for PDF loading and retrieval
│   ├── weather.py       # helper functions for weather retrieval
│   └── ui_app.py        # Streamlit user interface
└── tests/
    ├── test_api.py      # tests OpenWeatherMap API wrapper
    ├── test_rag.py      # tests PDF loading and retrieval
    └── test_graph.py    # tests decision logic and pipeline
```

## Setup

1. **Clone the repository** and navigate into it:

```bash
git clone https://github.com/<your-username>/langgraph-rag-weather-agent.git
cd langgraph-rag-weather-agent
```

2. **Create a virtual environment** and install dependencies.  The project requires Python 3.10.13.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables**.  Copy `.env.example` to `.env` and fill in your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
QDRANT_URL=   # leave empty for local mode
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag-weather-agent
```

The application uses `load_dotenv` to read these values at runtime.

4. **Run the Streamlit app**:

```bash
streamlit run src/ui_app.py
```

This will start a local web server where you can ask questions about the PDF or inquire about the weather in a city.

5. **Run tests**:

```bash
pytest -q
```

## Implementation overview

### RAG pipeline

The RAG component follows a typical four‑step workflow【190255345688846†L42-L56】:

1. **Split the PDF into chunks** using a `RecursiveCharacterTextSplitter` to ensure that each chunk is of a manageable size.
2. **Generate embeddings** for each chunk using `OpenAIEmbeddings` (or another embedding model) and store them in a Qdrant collectio.
3. **Retrieve relevant chunks** from Qdrant via a similarity search based on the user’s query.
4. **Generate an answer** by feeding the retrieved context and the question into an LLM.

### Decision node

LangGraph is used to orchestrate the pipeline.  The `decision_node` calls a ChatOpenAI model to classify the user question.  If the model responds with “weather” the graph routes to the weather node; otherwise it routes to the RAG node.  LangGraph’s conditional edges allow the agent to decide which data source to use for each query.

### Weather retrieval

When the decision node selects the weather path, the `weather_node` extracts a city name from the question and queries the OpenWeatherMap API.  The API call requires latitude, longitude and an API key and returns the current weather conditions in JSON format.  The response is summarised and returned as context for the final answer.

### Evaluation with LangSmith

LangSmith provides tooling to trace and evaluate LLM applications.  The project includes an example script showing how to set up a dataset and run an evaluation using the `evaluate()` method.  You can upload evaluation datasets, define evaluators and view experiment results in the LangSmith UI.

### Streamlit UI

The Streamlit interface uses `st.chat_input()` to capture user input and `st.chat_message()` to display messages.  Each time the user enters a question, the app invokes the LangGraph pipeline and displays the assistant’s response in a chat‑like format.

## Next steps

This repository provides a skeleton implementation that can be extended in many ways.  You might consider adding better entity extraction for weather queries, supporting multiple documents, experimenting with different embedding models or improving the UI.  LangGraph and LangChain are flexible frameworks that allow you to build more sophisticated agentic workflows, and Qdrant scales seamlessly as your data grows.