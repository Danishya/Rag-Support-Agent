"""
agent.py
The core of the project: an agent that decides, for each user question,
whether to (a) answer from the RAG knowledge base, or (b) call the
create_support_ticket tool, or both. Uses a local model via Ollama so
there is zero API cost.

Run interactively from the terminal:
    python agent.py

Prerequisites:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:    ollama pull llama3.1
    3. Run ingest.py first to build the vector store.
"""

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import create_support_ticket

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1"  # change to whatever you `ollama pull`ed


def build_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3})


retriever = build_retriever()


@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the internal IT support knowledge base for relevant information.
    Use this FIRST for any factual question about procedures, troubleshooting
    steps, or policies. Returns the most relevant document excerpts along with
    their source file, or a message saying nothing relevant was found.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the knowledge base."

    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_agent():
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    system_prompt = """You are an IT support assistant. For every user question:

1. ALWAYS call search_knowledge_base first to check for relevant documented
   procedures before answering.
2. Answer using ONLY the retrieved information. If the knowledge base has no
   relevant information, say so honestly - do not make up an answer.
3. Always mention which source file the information came from.
4. If the issue is a security concern, cannot be resolved from the documented
   steps, or the user explicitly asks to escalate/log a ticket, call
   create_support_ticket with a clear summary and appropriate priority.
5. Be concise and practical, like a real support agent would be."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    tools = [search_knowledge_base, create_support_ticket]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    print("IT Support Agent (Ollama + RAG). Type 'quit' to exit.\n")
    executor = build_agent()

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        result = executor.invoke({"input": query})
        print(f"\nAgent: {result['output']}\n")


if __name__ == "__main__":
    main()
