from score import get_response, stream_output, init
from fastapi import Request,FastAPI
from pydantic import BaseModel
import uvicorn

from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


class Prompt(BaseModel):
    prompt: str

model, tokenizer = init()

@app.get('/hello')
async def home():
    return {"message": "Hello World"}

@app.get('/liveness')
async def liveness():
    return {"message": "alive"}

@app.get('/readiness')
async def readiness():
    return {"message": "ready"}

@app.post("/prompt")
async def getsummary(user_request_in: Prompt):
    output_stream = get_response(user_request_in.prompt, model, tokenizer, torch_device)
    outputs = stream_output(output_stream)
    return outputs

