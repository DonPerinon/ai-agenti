from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1", temperature=0)



# Prompt
members = ["researcher", "coder"]
system_prompt = (
    "You are the speacker node your only task is to process user input and aswer in most polite way"
)

options = ["__end__"] + members
prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

speaker_chain = (
    prompt | llm | StrOutputParser()
)
