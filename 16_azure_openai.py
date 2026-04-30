"""
============================================================
AI-102 | Program 16 — Azure OpenAI Service
Service : Azure OpenAI
Skill   : Develop generative AI solutions
============================================================
Features:
  • Chat completions (GPT-4o, GPT-4, GPT-3.5-turbo)
  • System prompts & prompt engineering
  • Streaming responses
  • Function calling / tool use
  • Embeddings (text-embedding-ada-002)
  • DALL-E image generation
  • JSON mode (structured output)
============================================================
"""

import os
import json
from openai import AzureOpenAI

AZURE_OAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "https://<your-resource>.openai.azure.com/")
AZURE_OAI_KEY        = os.getenv("AZURE_OPENAI_KEY", "<your-oai-key>")
AZURE_OAI_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
CHAT_DEPLOYMENT      = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
EMBED_DEPLOYMENT     = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-ada-002")
DALLE_DEPLOYMENT     = os.getenv("AZURE_OPENAI_DALLE_DEPLOYMENT", "dall-e-3")

def get_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_OAI_ENDPOINT,
        api_key=AZURE_OAI_KEY,
        api_version=AZURE_OAI_API_VER
    )

# ── 1. Basic Chat Completion ──────────────────────────────
def chat_completion(user_message: str, system_prompt: str = None) -> str:
    """
    Basic chat completion with optional system prompt.
    system_prompt controls model behaviour and persona.
    """
    client = get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        temperature=0.7,        # 0=deterministic, 1=creative
        max_tokens=500,
        top_p=0.95,
    )

    reply = response.choices[0].message.content

    print("\n" + "="*65)
    print("  CHAT COMPLETION")
    print("="*65)
    print(f"  System : {system_prompt[:60] if system_prompt else 'None'}")
    print(f"  User   : {user_message[:60]}")
    print(f"  Reply  : {reply[:300]}")
    print(f"\n  Usage  : prompt={response.usage.prompt_tokens} "
          f"completion={response.usage.completion_tokens} "
          f"total={response.usage.total_tokens}")
    return reply

# ── 2. Multi-Turn Conversation ────────────────────────────
def multi_turn_chat(conversation_history: list[dict], new_message: str) -> tuple[str, list]:
    """
    Continue a multi-turn conversation.
    Maintains history for context continuity.
    """
    client = get_client()

    conversation_history.append({"role": "user", "content": new_message})

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=conversation_history,
        temperature=0.7,
        max_tokens=500,
    )

    assistant_reply = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply, conversation_history

# ── 3. Streaming Response ─────────────────────────────────
def chat_streaming(user_message: str) -> str:
    """
    Stream response tokens as they are generated.
    Better UX for long responses.
    """
    client = get_client()

    print("\n" + "="*65)
    print("  STREAMING CHAT COMPLETION")
    print("="*65)
    print(f"  User: {user_message}")
    print("  AI  : ", end="", flush=True)

    full_response = ""
    stream = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": user_message}],
        stream=True,        # Enable streaming
        max_tokens=300,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            full_response += token

    print()  # New line after streaming
    return full_response

# ── 4. Function Calling (Tool Use) ────────────────────────
def function_calling_demo(user_message: str) -> None:
    """
    Define functions the model can call to get real data.
    Model decides WHEN to call the function and WITH WHAT args.
    """
    client = get_client()

    # Define available functions/tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name e.g. Dar es Salaam"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_courses",
                "description": "Search for training courses by topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]}
                    },
                    "required": ["topic"]
                }
            }
        }
    ]

    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=messages,
        tools=tools,
        tool_choice="auto"    # 'auto', 'none', or specific function
    )

    choice = response.choices[0]

    print("\n" + "="*65)
    print("  FUNCTION CALLING")
    print("="*65)
    print(f"  User message: {user_message}")
    print(f"  Finish reason: {choice.finish_reason}")

    if choice.finish_reason == "tool_calls":
        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"\n  Model called: {func_name}()")
            print(f"  Arguments   : {func_args}")

            # Simulate function execution
            if func_name == "get_current_weather":
                func_result = f"27°C, partly cloudy in {func_args['location']}"
            elif func_name == "search_courses":
                func_result = f"Found 5 {func_args['topic']} courses at {func_args.get('level','all')} level"
            else:
                func_result = "Function not implemented"

            # Send function result back to model
            messages.append(choice.message)
            messages.append({
                "role": "tool",
                "content": func_result,
                "tool_call_id": tool_call.id
            })

        final_response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=messages,
        )
        print(f"\n  Final reply: {final_response.choices[0].message.content}")

# ── 5. JSON Mode (Structured Output) ─────────────────────
def json_mode_response(prompt: str) -> dict:
    """
    Force the model to return valid JSON.
    Set response_format={"type": "json_object"}.
    """
    client = get_client()

    response = client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},  # Force JSON output
        temperature=0,
        max_tokens=500,
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    print("\n" + "="*65)
    print("  JSON MODE OUTPUT")
    print("="*65)
    print(f"  Prompt: {prompt}")
    print(f"  JSON  : {json.dumps(parsed, indent=2)}")
    return parsed

# ── 6. Embeddings ─────────────────────────────────────────
def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert text to numerical vector representations.
    Used for: semantic search, similarity, clustering, RAG.
    """
    client = get_client()

    response = client.embeddings.create(
        model=EMBED_DEPLOYMENT,
        input=texts
    )

    print("\n" + "="*65)
    print("  EMBEDDINGS")
    print("="*65)

    embeddings = []
    for i, item in enumerate(response.data):
        embedding = item.embedding
        embeddings.append(embedding)
        print(f"  Text: '{texts[i][:50]}'")
        print(f"  Vector dimensions: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print()

    # Compute cosine similarity between first two
    if len(embeddings) >= 2:
        import math
        def cosine_sim(a, b):
            dot = sum(x*y for x,y in zip(a,b))
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            return dot / (mag_a * mag_b)

        sim = cosine_sim(embeddings[0], embeddings[1])
        print(f"  Cosine similarity between text 1 and 2: {sim:.4f}")

    return embeddings

# ── 7. DALL-E Image Generation ────────────────────────────
def generate_image(prompt: str) -> str:
    """
    Generate an image from a text prompt using DALL-E 3.
    Returns the URL of the generated image.
    """
    client = get_client()

    response = client.images.generate(
        model=DALLE_DEPLOYMENT,
        prompt=prompt,
        n=1,                         # Number of images (1 for DALL-E 3)
        size="1024x1024",            # 1024x1024, 1792x1024, 1024x1792
        quality="standard",          # 'standard' or 'hd'
        style="vivid"                # 'vivid' or 'natural'
    )

    image_url = response.data[0].url
    revised_prompt = response.data[0].revised_prompt

    print("\n" + "="*65)
    print("  DALL-E IMAGE GENERATION")
    print("="*65)
    print(f"  Original Prompt : {prompt}")
    print(f"  Revised Prompt  : {revised_prompt[:150]}")
    print(f"  Generated URL   : {image_url[:100]}")
    return image_url

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    system = "You are an expert cybersecurity trainer specializing in Azure AI security."

    # 1. Basic chat
    chat_completion(
        user_message="What are the top 3 cybersecurity threats in 2025?",
        system_prompt=system
    )

    # 2. Multi-turn
    history = [{"role": "system", "content": system}]
    reply1, history = multi_turn_chat(history, "What is zero trust security?")
    reply2, history = multi_turn_chat(history, "How does Azure implement this?")
    print(f"\n  Multi-turn reply 2: {reply2[:200]}")

    # 3. Streaming
    chat_streaming("Explain Azure AI Foundry in 3 sentences")

    # 4. Function calling
    function_calling_demo("What is the weather like in Dar es Salaam?")

    # 5. JSON mode
    json_mode_response(
        "List 3 Azure AI services as JSON: {services: [{name, purpose, sdk_package}]}"
    )

    # 6. Embeddings
    generate_embeddings([
        "Azure AI is a cloud AI platform",
        "Microsoft Azure provides machine learning services",
        "Cybersecurity protects digital assets",
    ])

    # 7. DALL-E
    generate_image("A futuristic cybersecurity operations centre in Tanzania, digital art style")

    print("\n  KEY POINTS FOR AI-102:")
    print("  • AzureOpenAI client needs endpoint + key + api_version")
    print("  • model= is the DEPLOYMENT NAME, not model name")
    print("  • temperature=0 deterministic, =1 creative")
    print("  • stream=True for token-by-token output")
    print("  • tool_choice='auto' lets model decide when to call functions")
    print("  • response_format={'type':'json_object'} forces valid JSON")
    print("  • Embeddings: text-embedding-ada-002 = 1536 dimensions")
    print("  • DALL-E 3: n=1 only, sizes: 1024x1024/1792x1024/1024x1792")
    print("="*65 + "\n")
