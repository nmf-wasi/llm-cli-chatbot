# LLM Tool-Calling Agent (plain Python, no framework)

A small command-line chatbot built on the Gemini API (`google-genai` SDK) that can call Python functions as tools. It was written to understand what agent frameworks hide, so the tool loop, streaming, retries and error handling are all implemented by hand in a single file.

## Features

- **Manual function calling:** the SDK's automatic calling is turned off, so every step of the loop is visible in the code
- **Streaming:** text is printed as it arrives, including on turns that involve tool calls
- **Round cap:** the tool loop is limited to `MAX_ROUNDS` so a misbehaving model can't loop forever
- **Retry with backoff:** only for 429, 500 and 503 errors; other errors (bad key, wrong model name, bad request) fail immediately
- **Tool error handling:** a tool that crashes, or a call to an unknown tool, is reported back to the model as an error instead of crashing the program
- **Truncation warning:** warns when a response ends because of `MAX_TOKENS`
- **System instruction** to keep answers short and make the model use tools instead of guessing

## Tools

| Tool | What it does |
|---|---|
| `get_weather(city)` | Returns a temperature for a city (**fake data** for a few cities, used to practice the mechanics) |
| `add_number(a, b)` | Adds two numbers |

## Setup

```bash
git clone <https://github.com/nmf-wasi/llm-cli-chatbot>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your Gemini API key in .env
python llm_agent.py
```

Type `quit` or `exit` to leave. Get a free API key from Google AI Studio.

## Example

```
you> What's the temperature in Dublin plus 5?

[MODEL WANTS] get_weather ({'city': 'Dublin'})
[RESULT] {'city': 'Dublin', 'temp': 15}

[MODEL WANTS] add_number ({'a': 15, 'b': 5})
[RESULT] 20
Dublin is 15°C, and plus 5 that is 20.
```

## How the loop works

```
main(): loop: read input
   └─ run_conversation(user_input)
        ├─ stream_turn(user_input)         → prints text, returns any tool calls
        └─ up to MAX_ROUNDS times:
             if no calls: stop
             run_tool(call) for each call  → results
             stream_turn(results)          → prints text, returns new calls
```

The model can't run code. It can only reply with text or with a request to call a function. When it asks for a tool, the program runs the matching Python function and sends the result back to the model. The result goes on the `user` side of the conversation, because it is the client, not the model, that produced it. The model then either answers in text, which ends the loop, or asks for another tool. Tool calls are collected until the stream finishes, because a single turn can contain several calls and all their results must be sent back together. The `Chat` object stores the history, so each message only needs the new content.

## Project layout

```
llm_agent.py       the whole agent
requirements.txt   dependencies
.env.example       template for the API key
```

## Known limitations

- If a retry happens in the middle of a streamed response, partial text can be printed twice
- The tools use hard-coded fake data
- Conversation history grows without limit, so very long sessions will get slower and more expensive
- Only tested with Gemini Flash-Lite models