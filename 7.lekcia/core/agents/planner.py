from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")  # or whichever model you use




prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Your job is to create a step-by-step plan to accomplish the user's task.
            Guidelines for creating a plan: 
            1. Break down the task into small, actionable steps.
            
            If you or any of the other assistants have the final answer or deliverable, prefix your response with [[[[FINAL ANSWER]]]] so the team knows to stop.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
planner = create_react_agent(
    model=llm,
    tools=[], 
    name="Planner",
    prompt=prompt
)