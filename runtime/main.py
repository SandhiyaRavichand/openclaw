import os
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = NVIDIA_API_KEY
)

@app.get("/")
def home():
  return {"status": "openclaw runtime working"}

@app.post("/task")
async def task(data: dict):
  prompt = data["prompt"]

  completion = client.chat.completions.create(
    model="qwen/qwen3-coder-480b-a35b-instruct",
    messages=[{"role":"user","content":""}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=4096,
    # stream=True
  )

  result = completion.choices[0].message.content

  return {"response": result}
