from agents.researcher import initReseachernode
async def researcher_node_logic(state):
    researcher= await initReseachernode()
    result=await researcher.ainvoke(state)
    last_message = result["messages"][-1]
    return {
          "messages": [last_message],
    }