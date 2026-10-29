import os
import sys
import json
import asyncio
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from create_vector import build_vector_store

load_dotenv()

st.set_page_config(page_title="Gemini MCP + RAG Agent", layout="centered")
st.title("📁 Gemini MCP + RAG Document Agent")

os.makedirs("./assets", exist_ok=True)

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file", type=["pdf", "txt", "docx", "csv", "xlsx"]
)

if uploaded_file is not None:
    file_path = os.path.join("./assets", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing document and generating vector embeddings..."):
        build_vector_store(file_path)
    st.sidebar.success(f"Successfully indexed: {uploaded_file.name}")

user_input = st.text_input("Ask a question about your uploaded document:")

async def run_mcp_query(query: str):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    server = StdioServerParameters(
        command="python",
        args=['server.py']
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            mcp_tools = await session.list_tools()
            gemini_tools = []
            for tool in mcp_tools.tools:
                gemini_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=tool.name,
                                description=tool.description,
                                parameters=tool.inputSchema
                            )
                        ]
                    )
                )

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=query,
                config=types.GenerateContentConfig(
                    tools=gemini_tools
                )
            )

            if response.function_calls:
                function_call = response.function_calls[0]
                tool_name = function_call.name
                tool_args = dict(function_call.args)

                result = await session.call_tool(tool_name, tool_args)
                rag_content = result.content[0].text

                rag_prompt = f"""
                    - Context: {rag_content}
                    - Question: {query}
                    - Please use the above context to answer only.
                """
                final_response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=rag_prompt
                )
                return final_response.text, rag_content, tool_name
            else:
                return "Sorry, I can only provide information based on the uploaded document.", "", None


if st.button("Submit Query") and user_input:
    if not os.path.exists("./vectorstore"):
        st.warning("Please upload a document first using the sidebar.")
    else:
        with st.spinner("Querying agent via MCP & RAG..."):
            try:
                answer, context, tool_used = asyncio.run(run_mcp_query(user_input))
                
                st.subheader("AI Response")
                st.write(answer)
                
                # if context:
                #     with st.expander("View Retrieved Context (RAG)"):
                #         st.text(context)
                #         st.caption(f"Tool executed: {tool_used}")
            except Exception as e:
                st.error(f"An error occurred: {e}")