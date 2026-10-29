import os
from mcp.server import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("RAG+MCP server")

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-embedding-001"
)

vector_store = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

@mcp.tool()
def search_doc(question: str) -> str:
    """
    Search the EjobIndia PDF.

    You MUST use this tool when the user asks about information contained in the uploaded document
    """

    docs = vector_store.similarity_search(
        query=question,
        k=2
    )
    context = "\n".join(doc.page_content for doc in docs)
    return context

if __name__ == "__main__":
    mcp.run()