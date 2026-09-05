"""
Nemotron agent client with tool calling support.
Uses OpenAI SDK pointed at NVIDIA's API endpoint.
"""
import json
from openai import OpenAI
import config
from prompts import SYSTEM_PROMPT

client = OpenAI(
    base_url=config.NEMOTRON_BASE_URL,
    api_key=config.NVIDIA_API_KEY
)

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "capture_and_describe",
            "description": (
                "Captures a photo from the user's webcam and analyzes it using "
                "a vision AI model. Returns a detailed text description of the "
                "user's current surroundings including obstacles, people, vehicles, "
                "signs, terrain, crosswalks, doors, and anything else relevant to "
                "a visually impaired person navigating the physical world. Use this "
                "tool whenever the user asks about what is around them, in front of "
                "them, or needs visual information about their environment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": (
                            "Optional focus area for the description. Examples: "
                            "'obstacles', 'text and signs', 'people', 'traffic'. "
                            "If not provided, gives a general surroundings description."
                        )
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": (
                "Gets step-by-step walking directions from the user's current "
                "location to a specified destination. Returns a list of navigation "
                "steps with distances. Use this when the user asks to go somewhere, "
                "requests directions, or mentions a destination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "The destination address or place name."
                    },
                    "origin": {
                        "type": "string",
                        "description": (
                            "The starting location. If not provided, uses the "
                            "user's last stated location."
                        )
                    }
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_continuous_mode",
            "description": (
                "Toggles continuous navigation mode on or off. When enabled, the "
                "system periodically captures webcam frames and proactively alerts "
                "the user about important changes in their surroundings. Use this "
                "when the user asks to start or stop continuous guidance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to enable, false to disable."
                    },
                    "interval_seconds": {
                        "type": "number",
                        "description": "Seconds between captures. Default 5."
                    }
                },
                "required": ["enabled"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": (
                "Captures a photo and reads all visible text — signs, labels, "
                "menus, documents, screens, etc. Use when the user asks 'what "
                "does that sign say', 'read that for me', 'is there any text'."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def call_nemotron(conversation_history: list[dict]) -> dict:
    """
    Send conversation history to Nemotron and get a response.
    Returns the full response choice (may contain tool_calls or text content).
    """
    response = client.chat.completions.create(
        model=config.NEMOTRON_MODEL,
        messages=conversation_history,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        temperature=config.NEMOTRON_TEMPERATURE,
        max_tokens=config.NEMOTRON_VISION_ONLY_MAX_TOKENS if config.VISION_ONLY_MODE else config.NEMOTRON_MAX_TOKENS,
    )
    return response.choices[0]


def run_agentic_loop(conversation_history: list[dict], tool_handlers: dict) -> str:
    """
    Run the agentic loop: call Nemotron, handle tool calls, repeat until text response.

    Args:
        conversation_history: The full conversation (mutated in place).
        tool_handlers: Dict mapping tool names to callable functions.

    Returns:
        The final text response from Nemotron.
    """
    max_iterations = config.AGENTIC_LOOP_MAX_ITERATIONS

    for _ in range(max_iterations):
        choice = call_nemotron(conversation_history)

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            assistant_message = choice.message.content or ""
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            return assistant_message

        # Nemotron wants to call tools
        conversation_history.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in choice.message.tool_calls
            ]
        })

        for tool_call in choice.message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            if func_name in tool_handlers:
                try:
                    result = tool_handlers[func_name](**func_args)
                except Exception as e:
                    result = f"Error executing {func_name}: {str(e)}"
            else:
                result = f"Unknown tool: {func_name}"

            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return "I'm having trouble processing that request. Could you try again?"


def create_initial_history() -> list[dict]:
    """Create a fresh conversation history with the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]
