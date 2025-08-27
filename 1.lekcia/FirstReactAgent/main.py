import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from functions.openai import call_openai
from classes.ReactAgent import ReactAgent


load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


conversation_history = [{"role": "system", "content": "You are helpfull assistant"}]

def main():
    agent = ReactAgent()
    while True:
        try:
            user_input = input("\n🤔 Your question: ").strip()
        
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using the Educational Database Tools AI Agent!")
                break
            conversation_history.append({"role": "user", "content": user_input})
            result = agent.run(conversation_history,client)
            conversation_history.append({"role":"assistant","content": result})

            print(conversation_history)
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Educational Database Tools AI Agent!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")



if __name__ == "__main__":
    main()
    
