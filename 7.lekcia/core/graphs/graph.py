from langgraph.graph import StateGraph, START, END

from nodes.researcher_node import researcher_node_logic
from nodes.speaker_node import speaker_node_logic
from nodes.supervisor_node  import supervisor_node_logic,supervisor_router
from shared.state import GraphState
from shared.visualizer import visualize

workflow = StateGraph(GraphState)

workflow.add_node("researcher", researcher_node_logic)
workflow.add_node("Supervisor", supervisor_node_logic)
workflow.add_node("speaker",speaker_node_logic), 

workflow.add_edge(START, "Supervisor")
workflow.add_conditional_edges(
    "Supervisor",
    supervisor_router,
    {"researcher": "researcher","speaker":"speaker"},
)
workflow.add_edge("researcher", "Supervisor")
workflow.add_edge("speaker",END)

trip_graph = workflow.compile()
visualize(trip_graph, "graph.png")