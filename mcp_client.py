import os
import asyncio
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


client = MultiServerMCPClient(
    {
        # Remote Tavily MCP Server
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },

        # Local AviationStack MCP Server
        "aviationstack": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY,
            },
        },

        # Local Weather MCP Server
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                os.path.join(
                    os.path.dirname(__file__),
                    "weather_mcp_server.py"
                )
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
            },
        },
    }
)


search_tool = None
aviation_tools = {}

# # Globally storing tavily_search mcp tools & aviation tools for reusability 
async def initialize_mcp():
    global search_tool
    global aviation_tools
    if search_tool is not None and aviation_tools:
        return
    tools = await client.get_tools() 
    print("\nAvailable MCP Tools:")
    for tool in tools:
        print(tool.name) 
    # tavily_search tools stored in memory for reusability
    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_research"
    )
    # aviation tools stored in memory for reusability
    aviation_tools = {
        tool.name : tool
        for tool in tools
        if tool.name != "tavily_research"
    }
    

async def tavily_mcp_search(query: str):
    await initialize_mcp()
    result = await search_tool.ainvoke(
        {
            "input": query
        }
    )
    return result

async def aviation_mcp_call(tool_name: str, tool_args: dict = None):
    print("Entering aviation_mcp_call")
    tools = await client.get_tools()
    tool = next(
        t for t in tools
        if t.name == tool_name
    )
    result = await tool.ainvoke(tool_args or {})
    return result

async def get_airports():
    await initialize_mcp()
    tool = aviation_tools.get("list_airports")
    if not tool:
        return "Airport tool unavailable"
    result = await tool.ainvoke({})
    return result


async def get_airlines():
    await initialize_mcp()
    tool = aviation_tools.get("list_airlines")
    if not tool:
        return "Airline tool unavailable"
    result = await tool.ainvoke({})
    return result

# configure custom weather MCP tools 
weather_tool = None
forecast_tool = None


async def initialize_weather_tools():

    global weather_tool, forecast_tool

    if weather_tool is not None:
        return

    tools = await client.get_tools()

    weather_tool = next(
        t for t in tools
        if t.name == "get_current_weather"
    )

    forecast_tool = next(
        t for t in tools
        if t.name == "get_forecast"
    )


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    return await weather_tool.ainvoke(
        {
            "city": city
        }
    )


async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    return await forecast_tool.ainvoke(
        {
            "city": city
        }
    )

# async def main():
#     tools = await client.get_tools()

#     # print all available tools (testing)

#     print("\nAvailable tools exposed by MCP servers : ")
#     for tool in tools:
#         print(tool.name)

#     # using tavily_search tool (invoking it)

#     search_tool = next(
#             tool
#             for tool in tools
#             if tool.name == "tavily_search"
#      )

#      result = await search_tool.ainvoke(
#             {
#                 "query": "Best hotels in Delhi"
#             }
#      )

# if __name__ == "__main__":
#     asyncio.run(main())

from langchain_groq import ChatGroq

# LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

###################################
# Destination Extractor

def extract_destination(query: str):

    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()



# async def test_client():

#     try:
#         tools = await asyncio.wait_for(
#             client.get_tools(),
#             timeout=20
#         )

#         print("\nSUCCESS")
#         print(f"{len(tools)} tools loaded\n")

#         for tool in tools:
#             print(tool.name)

#     except asyncio.TimeoutError:
#         print("\nTIMEOUT")
#         print("client.get_tools() never returned.")

#     except Exception as e:
#         print("\nERROR")
#         print(type(e).__name__)
#         print(e)


# if __name__ == "__main__":
#     asyncio.run(test_client())