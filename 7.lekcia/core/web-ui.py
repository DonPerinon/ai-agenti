from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from shared.state import GraphState
from graphs.graph import trip_graph
import asyncio

print("Starting web server...")
print(os.getenv("OPENAI_API_KEY"))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")  # folder containing index.html

# Global chat state
chat_history: GraphState = {
    "messages": [],
    "nextStep": "",
}

# Serve the browser page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# WebSocket endpoint for chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    while True:
        try:
            # Wait for client message
            data = await websocket.receive_text()
            if data.lower() in ["quit", "exit"]:
                await websocket.send_text("Goodbye!")
                continue  # Keep the connection open

            # Store user message
            chat_history["messages"].append(("user", data))

            # Run your async streaming logic
            result = trip_graph.astream(chat_history)
            last_step = None
            async for step in result:
                last_step = step
                # Optional: send partial results here just balast no needed for now
                # await websocket.send_text(partial_text) 

            # Extract last message
            if last_step:
                last_value = list(last_step.values())[-1]
                last_msg = last_value["messages"][-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    last_message = last_msg["content"]
                elif hasattr(last_msg, "content"):
                    last_message = last_msg.content
                else:
                    last_message = last_msg

                chat_history["messages"].append(("assistant", last_message))
                await websocket.send_text(last_message)

        except WebSocketDisconnect:
            print("WebSocket disconnected. Waiting for reconnect...")
            break  # Exit current loop; browser can reconnect
        except Exception as e:
            print(f"Error in WebSocket: {e}")
            await websocket.send_text(f"❌ Error: {e}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)