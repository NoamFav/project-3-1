from fastapi import FastAPI, Request
from pydantic import BaseModel
from processor import analyze_text

app = FastAPI()


class Transcript(BaseModel):
    text: str


@app.post("/analyze")
async def analyze(transcript: Transcript):
    result = analyze_text(transcript.text)
    return {"strategies": result}
