from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from LLMInference import LLMInference
from IngestAudio import AudioIngestor

app = FastAPI()
audio_ingestor = AudioIngestor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference = LLMInference()

class LoadModelRequest(BaseModel):
    model_name: str

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 256
    temperature: float = 0.7

class QueryResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"message": "LLM Inference API is running"}

@app.get("/api/models")
async def get_models():
    models = inference.get_available_models()
    categorized = inference.get_categorized_models()
    return {"models": models, "categorized_models": categorized, "current_model": inference.llm_name}

@app.get("/api/trending-models")
async def get_popular_models():
    import requests
    import re
    
    models = {
        "popular": []
    }
    
    try:
        response_popular = requests.get("https://ollama.com/library?sort=popular", timeout=5)
        matches_popular = re.findall(r'href="/library/([^/"]+)"', response_popular.text)
        seen_popular = set()
        for match in matches_popular:
            if match not in seen_popular and match != "library":
                seen_popular.add(match)
                models["popular"].append(match)
                if len(models["popular"]) >= 8:
                    break
                    
        if len(models["popular"]) > 0:
            return models
            
    except Exception as e:
        print(f"Failed to fetch models from ollama.com: {e}")
        pass

    fallback = [
        "llama3.2", "mistral", "qwen2.5:0.5b", "gemma2", "phi3", "deepseek-coder-v2"
    ]
    return {"popular": fallback}

class DeleteModelRequest(BaseModel):
    model_name: str

@app.post("/api/models/delete")
async def delete_model(request: DeleteModelRequest):
    try:
        inference.delete_model(request.model_name)
        return {"status": "success", "message": f"Model {request.model_name} deleted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/load")
async def load_model(request: LoadModelRequest):
    from fastapi.responses import StreamingResponse
    import json
    import ollama
    
    available_models = inference.get_available_models()
    if request.model_name not in available_models:
        def pull_stream():
            try:
                for progress in ollama.pull(request.model_name, stream=True):
                    progress_dict = vars(progress) if hasattr(progress, '__dict__') else progress if isinstance(progress, dict) else dict(progress)
                    yield json.dumps(progress_dict) + "\n"
                
                # After successful pull, load the model natively
                inference.llm_name = request.model_name
                print(f"Model '{request.model_name}' ready.")
                yield json.dumps({"status": "success", "message": f"Model {request.model_name} loaded and ready."}) + "\n"
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
                
        return StreamingResponse(pull_stream(), media_type="application/x-ndjson")
    else:
        try:
            inference.load_model(request.model_name)
            return {"status": "success", "message": f"Model {request.model_name} loaded and ready."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

@app.post("/api/models/unload")
async def unload_model():
    try:
        inference.unload_model()
        return {"status": "success", "message": "Model unloaded successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/generate", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    response_text = inference.generate(
        prompt=request.query,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return {"response": response_text}

@app.post("/api/audio/digest")
async def audio_digest(file : UploadFile = File(...)):
    file_location = f"temp_{file.name}"

    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    transcribe= audio_ingestor._transcribe_(file_location)

    prompt = """

    Summarize this file in bullet points
    {transcribe}

    
    """

    summary = inference.generate(prompt=prompt)

    return{
        "transcribe" : transcribe,
        "summary" : summary
    }


    