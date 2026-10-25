from google import genai
from google.genai import types, errors
from dotenv import load_dotenv
import time

load_dotenv()
client = genai.Client()
MODEL = "gemini-3.5-flash-lite"
MAX_ROUNDS = 5


# funcs
def get_weather(city: str) -> dict:
    # raise ValueError("boom") for testing
    data = {
        "seoul": 15,
        "dhaka": 28,
        "dublin": 15,
    }
    return {
        "city": city,
        "temp": data.get(city.lower(), 20),
    }


def add_number(a, b):
    return a + b


# func schemas for model
weather_decl = {
    "name": "get_weather",
    "description": "get current temp inc celcius for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Paris"},
        },
        "required": ["city"],
    },
}

add_number_decl = {
    "name": "add_number",
    "description": "Adds 2 numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "first number, e.g. 1",
            },
            "b": {
                "type": "number",
                "description": "second number, e.g. 2",
            },
        },
        "required": ["a", "b"],
    },
}

TOOLS = [types.Tool(function_declarations=[weather_decl, add_number_decl])]


def call_with_retry(fn, retry=3):
    for attempt in range(retry):
        try:
            return fn()
        except errors.APIError as e:
            print(f"[API ERROR] code {e.code}, message = {str(e.message)[:80]}")
            if e.code in (429, 500, 503) and attempt < retry - 1:
                wait = 2**attempt
                print(f"Failed, waiting {wait}s foor next attempt...")
                time.sleep(wait)

            else:
                raise


def run_tool(call):
    print(f"\n[MODEL WANTS] {call.name} ({call.args})")

    try:
        if call.name == "get_weather":
            result = get_weather(**call.args)
        elif call.name == "add_number":
            result = add_number(**call.args)
        else:
            result = {"error": f"Unknown function : {call.name}"}
            print(f"[WARN] unknown function: {call.name}")

    except Exception as e:
        result = {"error": str(e)}

    print(f"[RESULT] {result}")

    return types.Part.from_function_response(
        name=call.name, response={"result": result}
    )


def stream_turn(chat, message):
    calls = []
    finish_reason = None

    for chunk in chat.send_message_stream(message):
        if chunk.candidates and chunk.candidates[0].content:
            for part in chunk.candidates[0].content.parts or []:
                if part.text:
                    print(chunk.text, end="", flush=True)
                if part.function_call:
                    calls.append(part.function_call)

            if chunk.candidates[0].finish_reason:
                finish_reason = chunk.candidates[0].finish_reason

    print()
    return calls, finish_reason


def run_conersation(chat, user_input):
    calls, finish_reason = call_with_retry(lambda: stream_turn(chat, user_input))
    for _ in range(MAX_ROUNDS):
        if not calls:
            break
        parts = [run_tool(call) for call in calls]
        calls, finish_reason = call_with_retry(lambda: stream_turn(chat, parts))

    if calls:
        print("\n[WARN] hit MAX_ROUNDS the model was still asking for tools")
    if finish_reason == types.FinishReason.MAX_TOKENS:
        print(f"[WARN] response was cut of (MAX_TOKENS)")


def main():
    # config
    config = types.GenerateContentConfig(
        system_instruction="You are a concise assistant. Use the tools for weather and arithmatic operations instead of guessing",
        tools=TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        max_output_tokens=200,
    )
    # new chat
    chat = client.chats.create(model=MODEL, config=config)
    print("Hi! Start typing\n")
    while True:
        user_input = input("you> ")
        if user_input.lower() in ("quit", "exit"):
            break
        run_conersation(chat, user_input)


if __name__ == "__main__":
    main()
