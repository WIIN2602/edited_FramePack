from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller.framepack_controller import FramePackStream
app = FastAPI()

@app.get("/")
def home():
    return "OK"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    allow_credentials=True
)

app.include_router(FramePackStream)