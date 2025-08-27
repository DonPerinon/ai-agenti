from openai import OpenAI
from tools.definition import tools
def call_openai(input:str,client:OpenAI,):
    print("Callling ai")
    
    return client.chat.completions.create(
        model='gpt-4o',
        messages=input,
        tools=tools,
        tool_choice="auto"

    )