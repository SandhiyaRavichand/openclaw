import os
from fastapi import FastAPI
from openai import OpenAI
from uuid import uuid4
from fastapi.staticfiles import StaticFiles

app = FastAPI()

PROJECTS_DIR = "/tmp/projects"

os.makedirs(PROJECTS_DIR, exist_ok=True)

app.mount(
    "/projects",
    StaticFiles(directory=PROJECTS_DIR),
    name="projects"
)

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
  # print("RAW DATA:", data)
  
  generation_prompt = f"""
  Create a complete modern HTML landing page. 
  User request: {data["prompt"]} 
  Return only HTML code. """

  completion = client.chat.completions.create(
    model="qwen/qwen3-coder-480b-a35b-instruct",
    messages=[{"role":"user","content":generation_prompt}],
    temperature=0.7,
    top_p=0.8,
    max_tokens=4096,
    # stream=True
  )

  generated_html = completion.choices[0].message.content

  project_id = str(uuid4())[:8] 
  
  project_dir = f"/tmp/projects/{project_id}" 
  
  os.makedirs(project_dir, exist_ok=True) 
  
  file_path = f"{project_dir}/index.html" 
  
  with open(file_path, "w", encoding="utf-8") as f: f.write(generated_html)

  project_url = (
    f"http://bykqvtcpbjkiz6aqd8w0cwxi.161.97.122.156.sslip.io/"
    f"projects/{project_id}/index.html"
)

  # return {"response": result}
  return {
    "response": f"""

Project created successfully.

Project ID: {project_id}

File Saved:
{file_path}

Preview URL:
{project_url}
"""
}
