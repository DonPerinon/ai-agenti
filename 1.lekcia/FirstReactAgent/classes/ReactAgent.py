import os
import json

from typing import List, Dict, Any
from openai import OpenAI
from tools.definition import tools
from tools.mapper import available_functions
import tools.implementation
from functions.openai import call_openai

class ReactAgent:
    """
    A lightweight agent using a ReAct-style loop:
    - Queries the model
    - Executes any requested tools
    - Feeds results back until only text is returned
    """

    def __init__(self, model: str = "gpt-4o", max_rounds: int = 8):
        self.model = model
        self.max_rounds = max_rounds

    def interact(self, history: List[Dict[str, Any]], client: OpenAI) -> str:
        rounds = 0

        while rounds < self.max_rounds:
            rounds += 1
            print(f"\n>>> Round {rounds}")

            # Query the LLM
            reply = call_openai(history, client)
            message = reply.choices[0].message

            # If model asks for tools
            if getattr(message, "tool_calls", None):
                # Log the tool calls in the conversation
                history.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ],
                    }
                )

                # Execute each requested tool
                for call in message.tool_calls:
                    fn_name = call.function.name
                    fn_args = json.loads(call.function.arguments)
                    call_id = call.id

                    print(f"Calling: {fn_name} with {fn_args}")

                    # Look up and run the function
                    fn = available_functions[fn_name]
                    result = fn(**fn_args)

                    print(f"Result: {result}")

                    # Store tool response back in history
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fn_name,
                            "content": json.dumps(result),
                        }
                    )

                continue  # Go to next loop

            # If no tool calls, final output
            answer = message.content
            history.append({"role": "assistant", "content": answer})

            print(f"\nFinal Response: {answer}")
            return answer

        return "Error: Stopped after too many rounds without resolution."
