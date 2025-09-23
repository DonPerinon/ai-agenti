from agents.supervisor import executor
def executor_node_logic(state):
    result= executor.invoke(state)
    last_message = result["messages"][-1]
    print("Last message:", last_message)
    return {
          "messages": [last_message],
    }