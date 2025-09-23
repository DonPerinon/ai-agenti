import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
load_dotenv()

from shared.state import GraphState
from graphs.graph import trip_graph
async def main():

    messages = [] # <--------------------------------------------------------- Store chat history
    initial_state: GraphState = {
    "messages": [],

    "nextStep": "",

    }

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            print(initial_state)
            # Add user's message to the history
            initial_state["messages"].append(("user", user_input))
            print(initial_state)
            result=trip_graph.astream(initial_state)
            last_step = None
            async for step in result:
                
                last_step = step

            # extract the last message
            if last_step:
                last_value = list(last_step.values())[-1]
                last_msg = last_value["messages"][-1]

                if isinstance(last_msg, dict) and "content" in last_msg:
                    last_message = last_msg["content"]
                elif hasattr(last_msg, "content"):
                    last_message = last_msg.content
                else:
                    last_message = last_msg  # string or fallback

                print("Assistant:", last_message)
                initial_state["messages"].append(("assistant", last_message)) 

        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Educational Database Tools AI Agent!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")





    


if __name__ == "__main__":
    asyncio.run(main())
