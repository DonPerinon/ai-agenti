from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser


# Tools = possible hand-offs
@tool("transfer_to_researcher", description="Ask the researcher more detail about requested products.")
def transfer_to_researcher():
    """Transfer the task to the researcher."""
    return {"next": "researcher"}

@tool("transfer_to_speaker", description="Trasfer message to user.")
def transfer_to_speaker():
    """Trasfer message to user."""
    return {"next": "speaker"}


tools = [transfer_to_researcher, transfer_to_speaker]

llm = ChatOpenAI(model="gpt-4.1", temperature=0)

llm_with_tools = llm.bind_tools(tools)


# Prompt
members = ["researcher", "coder", "speaker"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next only one at the time. Each worker will perform a"
    " task and respond with their results and status end marked with [[[[FINAL ANSWER]]]]. When finished,"
    " respond with FINISH."
)

options = ["__end__"] + members
prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {members}"),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt | llm_with_tools | StrOutputParser()
)
