from agents.dataEnrichment import de_chain
def de_node_logic(state):
    result=de_chain.invoke(state)
    print(result)

    return {
          "messages": [result],
    }