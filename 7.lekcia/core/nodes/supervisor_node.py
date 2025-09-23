from agents.supervisor import supervisor_chain
from typing import Literal

def supervisor_node_logic(state):
    print("supervisor")
    result=supervisor_chain.invoke(state)
    print(result)
    return {
          "nextStep": result,
    }

def supervisor_router(state) -> Literal["coder", "researcher", "FINISH"]:
    nextStep=state["nextStep"]
    return nextStep
