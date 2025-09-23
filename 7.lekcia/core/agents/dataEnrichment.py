from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1", temperature=0)



# Prompt
members = ["researcher", "coder"]
system_prompt = (
    "You are a data enrichment node. Your only task is to take the user message "
    "and make it clearer and more actionable. "
)

options = ["__end__"] + members
prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

de_chain = (
    prompt | llm | StrOutputParser()
)
