import os
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import Session
from google.adk.memory import InMemoryMemoryService
from google.genai import types # For creating message Content/Parts
from dotenv import load_dotenv


load_dotenv('./.env')

root_agent_model = "gemini-2.0-flash-exp"
greeting_agent_model = "gemini-2.0-flash"

session_service = InMemoryMemoryService() # Create a dedicated service

# Define constants for identifying the interaction context
APP_NAME = "First_Application_To_Test" # Unique app name for this test
USER_ID = "User_1"
SESSION_ID = "Session_001" # Using a fixed ID for simplicity

# Create the specific session where the conversation will happen
session = Session(
    app_name=APP_NAME,
    user_id=USER_ID,
    id=SESSION_ID
)

session = session_service.add_session_to_memory(
    session = session
)

def say_hello(name: str = "there") -> str:
    print(f"--- Tool: say_hello called with name: {name} ---")
    return f"Hello, {name}!"

def say_goodbye() -> str:
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."

def get_weather(city: str) -> dict:
    print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
    city_normalized = city.lower().replace(" ", "") # Basic normalization

    # Mock weather data
    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C. Light breeze from the southwest at 10 km/h. No rain expected today."
        },
        "london": {
            "status": "success",
            "report": "It's cloudy in London with a temperature of 15°C. Chance of light showers in the evening. Winds steady at 14 km/h from the west."
        },
        "tokyo": {
            "status": "success",
            "report": "Tokyo is experiencing light rain with a temperature of 18°C. Humidity is at 78% and rain expected to clear by late afternoon."
        },
        "paris": {
            "status": "success",
            "report": "Paris is partly cloudy with sunny intervals. Temperature is 20°C and mild winds at 8 km/h from the southeast."
        },
        "sydney": {
            "status": "success",
            "report": "It's a bright and clear day in Sydney with a high of 28°C. Perfect beach weather with UV index at 7 — sunscreen recommended!"
        },
        "mumbai": {
            "status": "success",
            "report": "Mumbai is hot and humid today, with a temperature of 33°C. Expect a light drizzle in the evening and 85% humidity throughout the day."
        },
        "berlin": {
            "status": "success",
            "report": "Berlin is cool and breezy, 12°C with scattered clouds. Winds blowing at 20 km/h from the northeast."
        },
        "cairo": {
            "status": "success",
            "report": "Cairo is sunny and dry, with a temperature of 35°C. Visibility is excellent and no precipitation is expected."
        }
    }
    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}

def summarize_article(text: str) -> dict:
    print(f"--- Tool: summarize_article called ---")
    # Mock summary
    summary = f"Summary: {text[:60]}..."  # Simulate summarizing
    return {"status": "success", "summary": summary}

def get_joke(category: str = "general") -> dict:
    print(f"--- Tool: get_joke called with category: {category} ---")
    mock_jokes = {
        "general": "Why don’t scientists trust atoms? Because they make up everything!",
        "tech": "Why do programmers prefer dark mode? Because light attracts bugs!",
        "dad": "I only know 25 letters of the alphabet. I don't know y.",
        "animal": "Why don’t seagulls fly over the bay? Because then they’d be bagels!",
        "math": "Why was the equal sign so humble? Because it knew it wasn’t less than or greater than anyone else.",
        "physics": "Schrödinger’s cat walks into a bar... and doesn’t.",
        "office": "Why did the scarecrow get promoted? Because he was outstanding in his field.",
        "coffee": "What did the coffee say to the sugar? You make life sweet!",
        "school": "Why was the math book sad? Because it had too many problems.",
        "developer": "There are only two hard things in Computer Science: cache invalidation, naming things, and off-by-one errors."
    }
    return {"status": "success", "joke": mock_jokes.get(category.lower(), mock_jokes["general"])}

greeting_agent = Agent(
        # Using a potentially different/cheaper model for a simple task
        model=greeting_agent_model,
        name="greeting_agent",
        instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                    "Use the 'say_hello' tool to generate the greeting. "
                    "If the user provides their name, make sure to pass it to the tool. "
                    "Do not engage in any other conversation or tasks.",
        description="Handles simple greetings and hellos using the 'say_hello' tool.", # Crucial for delegation
        tools=[say_hello],
    )

farewell_agent = Agent(
    # Can use the same or a different model
    model=greeting_agent_model, # Sticking with GPT for this example
    name="farewell_agent",
    instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
                "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
                "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
                "Do not perform any other actions.",
    description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.", # Crucial for delegation
    tools=[say_goodbye],
)

weather_agent = Agent(
    model=greeting_agent_model,  # Reuse your cheaper/faster model for this.
    name="weather_agent",
    instruction="You are the Weather Agent. Your ONLY task is to answer weather queries using the 'get_weather' tool. "
                "When a user asks about the weather in a specific city, call this tool and provide the answer.",
    description="Handles weather queries.",
    tools=[get_weather],
)

root_agent = Agent(
    name = "Root_Agent",
    model = root_agent_model,
    description="The main coordinator agent. Handles weather, summarization, jokes, and delegates greetings/farewells.",
    instruction="""
    You are the Root Agent coordinating a team of specialists. Your responsibilities are:
    1. If the user asks for multiple pieces of information (e.g., weather AND a joke), handle each request separately.
    2. Provide weather information using the 'get_weather' tool for weather-related questions.
    3. Delegate greetings like 'Hi' or 'Hello' to 'greeting_agent'.
    4. Delegate farewells like 'Bye' or 'See you' to 'farewell_agent'.
    5. Delegate text summarization to 'summarize_agent' tool.
    6. Delegate joke requests to 'joke_agent' tool.
    """,
    tools=[get_weather,summarize_article,get_joke],
    sub_agents=[greeting_agent, farewell_agent]
)

agent = root_agent

runner_root = Runner(
    app_name=APP_NAME,
    session_service=session,
    agent=root_agent
)

