# LangGraph Multi-Agent Travel Booking System with Long-Term Memory

# main.py

import os
import json
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

# this was using tavily api rather than mcp
# from tools.tavily_tool import tavily_search 

import asyncio
from mcp_client import tavily_mcp_search

# this was using aviation api rather than mcp
# from tools.flight_tool import search_flights

from mcp_client import (
    tavily_mcp_search,
    get_airports,
    get_airlines,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    TavilySearchError,
    weather_mcp_search
)

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
)

# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results: str


# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport reference data (not a route or live schedule):
{airport_data}

Airline reference data (not proof that an airline serves this route):
{airline_data}

Give concise, truthful guidance:

1. Use a departure city only if the user explicitly provided one. Otherwise say
   the departure city is needed; never assume the traveler is departing from
   the United States or any other country.
2. Do not claim a route, airline, schedule, duration, or fare is verified by
   the supplied airport/airline reference data.
3. Do not invent current prices or flight availability. Label any general
   advice as an estimate and state that live flight options were not retrieved.
4. If the origin is missing, provide only destination-specific general advice
   and ask the user to provide their departure city for route guidance.
"""

# Flight Agent
def flight_agent(state: TravelState):
    
    ## this was aviation api call , now mcp will be used
    # query = state["user_query"]
    # flight_data = search_flights(query)
    # return {
    #     "flight_results": flight_data,
    #     "messages": [
    #         AIMessage(content=f"Flight results fetched")
    #     ],
    #     "llm_calls": state.get("llm_calls", 0) + 1
    # }
    
    ## now , use AVIATION MCP CALL
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:
        # get airports from list_airport tool
        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )
        # get airlines from list_airlines tool
        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )
        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])
        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    try:
        search_results = asyncio.run(tavily_mcp_search(query))
    except TavilySearchError as exc:
        return {
            "hotel_results": f"Hotel search unavailable: {exc}",
            "messages": [
                AIMessage(content="Hotel search unavailable")
            ],
        }

    response = llm.invoke([
        SystemMessage(
            content=(
                "Summarize hotel search results for a travel planner. Use only "
                "hotel names, areas, amenities, prices, and links present in "
                "the search results. Do not invent availability or prices. "
                "Keep it concise: at most 5 options, each with a short reason "
                "to consider it. Preserve useful source links and clearly say "
                "when the search results do not contain a requested detail."
            )
        ),
        HumanMessage(content=f"Trip request:\n{state['user_query']}\n\nSearch results:\n{search_results}"),
    ])

    return {
        "hotel_results": response.content,
        "messages": [
            AIMessage(content="Hotel search results summarized")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    def parse_weather_data(data):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data

    current = parse_weather_data(weather_data)
    forecast = parse_weather_data(forecast_data)
    if isinstance(current, dict) and "temperature_c" in current:
        weather_summary = (
            f"Current weather in {current.get('city', city)}: "
            f"{current['temperature_c']}°C, "
            f"feels like {current.get('feels_like_c', 'unknown')}°C; "
            f"{current.get('condition', 'conditions unavailable')}. "
            f"Humidity: {current.get('humidity', 'unknown')}%; "
            f"wind: {current.get('wind_speed', 'unknown')} m/s."
        )
    else:
        weather_summary = f"Current weather: {current}"

    if isinstance(forecast, dict) and isinstance(forecast.get("forecast"), list):
        forecast_summary = "\n".join(
            f"- {item.get('datetime', 'Time unavailable')}: "
            f"{item.get('temperature', 'Unknown')}°C, "
            f"{item.get('weather', 'conditions unavailable')}"
            for item in forecast["forecast"]
            if isinstance(item, dict)
        )
        if not forecast_summary:
            forecast_summary = "No forecast entries were returned."
    else:
        forecast_summary = f"Forecast unavailable: {forecast}"

    return {
        "weather_results": (
            f"**{city}**\n\n{weather_summary}\n\n"
            f"Next forecast entries (3-hour intervals, not a full-trip forecast):\n"
            f"{forecast_summary}"
        ),
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# Itinerary Agent
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a practical, concise travel plan using the user's request and the
    agent results below.

    Accuracy requirements:
    - Do not invent facts, bookings, flight schedules, prices, or live
      availability. Clearly label estimates and say when information was not
      retrieved.
    - Do not infer the traveler's departure city or country. If it is missing,
      say departure details need confirmation.
    - Prefer the hotel search results for accommodation suggestions and the
      weather agent's current/forecast data for weather. Do not replace these
      with generic or seasonal assumptions.
    - The weather forecast contains only the next few 3-hour entries; do not
      present it as a full forecast for the entire trip.
    - Preserve the user's dates, duration, destination, origin, and budget
      when provided. Do not silently change them.
    - Use clear Markdown headings and short, useful bullet points.

    Required sections: Trip overview, Flights, Accommodation, Weather, Day-by-
    day itinerary, Budget/booking notes. If a section lacks reliable source
    data, say so plainly instead of guessing.

    User request:
    {state['user_query']}

    Flight agent results:
    {state['flight_results']}

    Hotel search results:
    {state['hotel_results']}
    
    Weather results:
    {state['weather_results']}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# building the whole graph workflow of travel planning agents

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)


# Persistent connection so both CLI and Streamlit can share the compiled app
_conn = psycopg.connect(DATABASE_URL)
_conn.autocommit = True

# Persist graph state so conversations can resume across requests
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

# Compile the workflow into an executable application with persistent memory
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # config = {
    #     "configurable": {
    #         "thread_id": "user_rishav"
    #     }
    # }

    # Create a unique conversation thread for this CLI session
    import uuid
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)
