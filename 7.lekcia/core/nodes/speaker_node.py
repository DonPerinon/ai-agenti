from agents.speaker import speaker_chain
def speaker_node_logic(state):
   result = speaker_chain.invoke(state)
 
   return {
      "messages": [result]
   }