#!/usr/bin/env python3
"""
AI Travel Agent — powered by OpenAI (Azure OpenAI or direct OpenAI API).

Usage:
    python travel_agent.py                        # interactive chat
    python travel_agent.py --destination "Paris"  # quick destination query

Azure OpenAI setup (.env):
    AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
    AZURE_OPENAI_API_KEY=your_key
    AZURE_OPENAI_DEPLOYMENT=gpt-4o
    AZURE_OPENAI_API_VERSION=2024-12-01-preview

Direct OpenAI setup (.env):
    OPENAI_API_KEY=your_key
    OPENAI_MODEL=gpt-4o
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich import print as rprint

from tools import (
    search_flights,
    search_hotels,
    get_attractions,
    get_restaurants,
    create_itinerary,
    estimate_budget,
    get_travel_info,
    get_weather_info,
    get_packing_list,
)

load_dotenv()

# ── Client setup ─────────────────────────────────────────────────────────────

def build_client() -> tuple[AzureOpenAI | OpenAI, str]:
    """
    Return an OpenAI client and model/deployment name.
    Prefers Azure OpenAI if AZURE_OPENAI_ENDPOINT is set,
    otherwise falls back to the direct OpenAI API.
    """
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o").strip()
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()

    if azure_endpoint and azure_key:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
        )
        return client, azure_deployment

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        rprint("[bold red]Error:[/] No API credentials found.\n"
               "Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY for Azure OpenAI, or\n"
               "OPENAI_API_KEY for the direct OpenAI API.\n"
               "See .env.example for details.")
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
    return OpenAI(api_key=openai_key), model


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI travel agent with 20+ years of experience planning trips worldwide. \
You have deep knowledge of flights, hotels, attractions, local culture, visa requirements, weather patterns, \
and travel budgeting across every continent.

Your personality:
- Warm, enthusiastic, and genuinely excited about travel
- Precise and practical — you give actionable advice, not vague tips
- You anticipate follow-up questions and answer them proactively
- You always consider the traveller's budget, preferences, and traveller type

When helping users plan trips, ALWAYS use the available tools to provide real data:
- search_flights — when asked about flights or how to get to a destination
- search_hotels — when asked about accommodation
- get_attractions — for sightseeing and things to do
- get_restaurants — for dining recommendations
- create_itinerary — to build day-by-day plans
- estimate_budget — for cost planning
- get_travel_info — for visa, safety, currency, health, and etiquette info
- get_weather_info — for climate and best-time-to-visit questions
- get_packing_list — for packing advice

After receiving tool results, always:
1. Summarise key findings in friendly, readable prose
2. Highlight the top 2-3 recommendations with reasons
3. Add important caveats or tips the tool data doesn't capture
4. Offer to dive deeper into any aspect

Format responses with markdown headers, bullet points, and bold text for key info."""

# ── Tool definitions (OpenAI function-calling format) ────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two cities. Returns airline options with prices, durations, and details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city (e.g. 'New York', 'London')"},
                    "destination": {"type": "string", "description": "Destination city (e.g. 'Paris', 'Tokyo')"},
                    "departure_date": {"type": "string", "description": "Departure date YYYY-MM-DD"},
                    "return_date": {"type": "string", "description": "Return date YYYY-MM-DD (for round trips)"},
                    "passengers": {"type": "integer", "description": "Number of passengers (default 1)"},
                    "travel_class": {"type": "string", "enum": ["economy", "premium economy", "business", "first"], "description": "Cabin class"},
                },
                "required": ["origin", "destination", "departure_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search hotel accommodation at a destination across different price tiers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "City to search hotels in"},
                    "check_in": {"type": "string", "description": "Check-in date (YYYY-MM-DD)"},
                    "check_out": {"type": "string", "description": "Check-out date (YYYY-MM-DD)"},
                    "guests": {"type": "integer", "description": "Number of guests (default 2)"},
                    "budget_per_night": {"type": "number", "description": "Max budget per night in USD (optional)"},
                    "accommodation_type": {"type": "string", "description": "Type: hotel, boutique, hostel, villa"},
                },
                "required": ["destination", "check_in", "check_out"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_attractions",
            "description": "Get top attractions, must-see sights, and hidden gems for a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "City or destination"},
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Interests e.g. ['history', 'art', 'food', 'nature', 'adventure']",
                    },
                    "duration_days": {"type": "integer", "description": "Number of days at the destination"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_restaurants",
            "description": "Get restaurant and dining recommendations for a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "City for restaurant recommendations"},
                    "cuisine_type": {"type": "string", "description": "Preferred cuisine (e.g. 'Italian', 'local')"},
                    "budget_level": {"type": "string", "enum": ["budget", "mid_range", "upscale", "mixed"]},
                    "dietary_preferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dietary needs e.g. ['vegetarian', 'halal', 'gluten-free']",
                    },
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_itinerary",
            "description": "Create a detailed day-by-day travel itinerary for a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Travel destination"},
                    "duration_days": {"type": "integer", "description": "Number of days"},
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Traveller interests e.g. ['food', 'history', 'art', 'adventure']",
                    },
                    "traveler_type": {"type": "string", "enum": ["solo", "couple", "family", "group", "budget", "luxury"]},
                },
                "required": ["destination", "duration_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_budget",
            "description": "Estimate total trip cost including flights, accommodation, food, and activities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Travel destination"},
                    "duration_days": {"type": "integer", "description": "Length of trip in days"},
                    "travelers": {"type": "integer", "description": "Number of travellers (default 2)"},
                    "traveler_type": {"type": "string", "enum": ["budget", "mid_range", "luxury"]},
                    "include_flights": {"type": "boolean", "description": "Include round-trip flights (default true)"},
                    "home_country": {"type": "string", "description": "Traveller's home country (default USA)"},
                },
                "required": ["destination", "duration_days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_travel_info",
            "description": "Get visa requirements, currency, safety, health advice, electricity, emergency numbers, and cultural etiquette.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Travel destination"},
                    "home_country": {"type": "string", "description": "Traveller's passport country (e.g. 'usa', 'uk', 'india')"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_info",
            "description": "Get weather information, best time to visit, and monthly climate overview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Travel destination"},
                    "travel_month": {"type": "string", "description": "Month of travel e.g. 'January', 'July'"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_packing_list",
            "description": "Generate a tailored packing list based on destination, duration, and activities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Travel destination"},
                    "duration_days": {"type": "integer", "description": "Length of trip in days"},
                    "activities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Planned activities e.g. ['beach', 'city', 'adventure', 'cultural_sites']",
                    },
                    "season": {"type": "string", "description": "Season at destination: warm, cold, tropical, rainy"},
                },
                "required": ["destination", "duration_days"],
            },
        },
    },
]

# ── Tool dispatcher ───────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "get_attractions": get_attractions,
    "get_restaurants": get_restaurants,
    "create_itinerary": create_itinerary,
    "estimate_budget": estimate_budget,
    "get_travel_info": get_travel_info,
    "get_weather_info": get_weather_info,
    "get_packing_list": get_packing_list,
}


def execute_tool(name: str, arguments: str) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        tool_input = json.loads(arguments)
        return json.dumps(fn(**tool_input), indent=2, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── Core agent loop ───────────────────────────────────────────────────────────

def run_agent(
    client: AzureOpenAI | OpenAI,
    model: str,
    messages: list[dict],
    console: Console,
) -> str:
    """Run one turn of the agentic loop (handles multiple tool call rounds)."""
    while True:
        with console.status("[bold cyan]Thinking...[/]", spinner="dots"):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content or ""

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return choice.message.content or ""

        # Append the assistant message (with tool_calls) to history
        messages.append(choice.message.model_dump())

        # Execute each tool call and append results
        for tool_call in choice.message.tool_calls:
            name = tool_call.function.name
            console.print(f"  [dim]→ Calling tool:[/] [bold yellow]{name}[/]")
            result = execute_tool(name, tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


# ── UI helpers ────────────────────────────────────────────────────────────────

WELCOME = """
# ✈️  AI Travel Agent

Your personal travel expert — powered by **OpenAI** (Azure OpenAI or direct API).

I can help you with:
- **Flight search** — compare airlines, prices, and routes
- **Hotel recommendations** — across all budgets
- **Itinerary planning** — day-by-day schedules
- **Budget estimation** — full cost breakdown
- **Visa & entry requirements** — for your passport
- **Weather & best time to visit**
- **Top attractions & hidden gems**
- **Restaurant recommendations**
- **Packing lists** — tailored to your trip

Type **exit** or **quit** to end the session.
"""

EXAMPLE_PROMPTS = [
    'Plan a 7-day honeymoon in Paris for 2 people, budget around $5,000',
    'I want to visit Tokyo for 10 days next April. What should I know?',
    'Compare flights from New York to Bali in June for 2 people',
    'What are the visa requirements for visiting Dubai as a US citizen?',
    'Create a 5-day family itinerary for Rome with two young kids',
    'What is the best time of year to visit Bali?',
    'Help me plan a budget backpacking trip through Southeast Asia',
]


def interactive_chat(client: AzureOpenAI | OpenAI, model: str, backend: str) -> None:
    console = Console()
    console.print(Panel(Markdown(WELCOME), border_style="cyan", padding=(1, 2)))
    console.print(f"[dim]Backend: {backend}[/]\n")
    console.print("[bold]Example questions to get started:[/]")
    for i, p in enumerate(EXAMPLE_PROMPTS, 1):
        console.print(f"  [dim]{i}.[/] {p}")
    console.print()

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! Safe travels! ✈️[/]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye", "q"):
            console.print("[dim]Goodbye! Safe travels! ✈️[/]")
            break

        messages.append({"role": "user", "content": user_input})
        reply = run_agent(client, model, messages, console)

        console.print()
        console.print(Rule("[bold cyan]Travel Agent[/]"))
        console.print(Markdown(reply))
        console.print()

        messages.append({"role": "assistant", "content": reply})


def quick_query(client: AzureOpenAI | OpenAI, model: str, destination: str) -> None:
    console = Console()
    query = (
        f"Give me a comprehensive travel overview for {destination}: "
        "top attractions, best time to visit, rough budget estimate for 7 days "
        "(mid-range, 2 people), and the 3 most important things to know before going."
    )
    console.print(Panel(f"[bold]Quick overview:[/] {destination}", border_style="cyan"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    reply = run_agent(client, model, messages, console)
    console.print(Markdown(reply))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Travel Agent — OpenAI / Azure OpenAI")
    parser.add_argument("--destination", "-d", metavar="CITY",
                        help="Quick travel overview for a city then exit.")
    args = parser.parse_args()

    client, model = build_client()

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        backend = f"Azure OpenAI  ({endpoint}) · deployment: {model}"
    else:
        backend = f"OpenAI API (direct) · model: {model}"

    if args.destination:
        quick_query(client, model, args.destination)
    else:
        interactive_chat(client, model, backend)


if __name__ == "__main__":
    main()
