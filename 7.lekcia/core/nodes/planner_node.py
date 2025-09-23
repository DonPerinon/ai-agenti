from agents.planner import planner
def planner_node_logic(state):
    result= planner.invoke(state)
    print(result)
    last_message = result["messages"][-1]
    print("Last message:", last_message)
    return {
          "messages": [last_message],
    }