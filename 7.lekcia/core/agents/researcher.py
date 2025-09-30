import os
from langgraph_supervisor import create_supervisor, create_handoff_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4.1",temperature=0)  # or whichever model you use

MCP_SERVER = os.getenv("MCP_SERVER_URL")
print(MCP_SERVER)
client = MultiServerMCPClient(
    {
        "google": {
            # Ensure you start mcp server before running this code
            "url": MCP_SERVER, 
            "transport": "streamable_http",
        }
    }
)

async def initReseachernode():
    tools = await client.get_tools()
    # Planner: turns request into high-level itinerary

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Your job is to find most relevant info about the product or set of products user defined
                Guidelines for creating a research:
                1. Iterate over tools list
                2. prefix your response with [[[[FINAL ANSWER]]]] so the team knows to stop
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    researcher = create_react_agent(
        model=llm,
        tools=tools, 
        name="Researcher",
        prompt=prompt
    )
    return researcher