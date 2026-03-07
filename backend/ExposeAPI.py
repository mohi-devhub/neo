import os
import json
import re
import tempfile
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import ollama
import pdfplumber
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Literal

import Database as db
from EmbedData import EmbeddingPipeline
from IngestAudio import AudioIngestor
from IngestDocs import DocumentIngestor
from IngestVideo import VideoIngestor
from LLMInference import LLMInference
from VectorStore import VectorStore


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), ".registry.json")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
TEXT_EXTENSIONS  = {".txt", ".md"}
PDF_EXTENSIONS   = {".pdf"}

inference      = LLMInference()
audio_ingestor = AudioIngestor()
doc_ingestor   = DocumentIngestor()
video_ingestor = VideoIngestor()


# ---------------------------------------------------------------------------
# Vector store cache
# ---------------------------------------------------------------------------

_vs_cache: dict = {}

def get_vector_store(session_id: Optional[int]) -> VectorStore:
    key = session_id if session_id is not None else "global"
    if key not in _vs_cache:
        if session_id is None:
            idx  = os.path.join(_DATA_DIR, "faiss_index.bin")
            meta = os.path.join(_DATA_DIR, "metadata.json")
        else:
            idx  = os.path.join(_DATA_DIR, f"faiss_{session_id}.bin")
            meta = os.path.join(_DATA_DIR, f"faiss_{session_id}_meta.json")
        _vs_cache[key] = VectorStore(index_path=idx, metadata_path=meta)
    return _vs_cache[key]

def delete_vector_store(session_id: int) -> None:
    _vs_cache.pop(session_id, None)
    for path in [
        os.path.join(_DATA_DIR, f"faiss_{session_id}.bin"),
        os.path.join(_DATA_DIR, f"faiss_{session_id}_meta.json"),
    ]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SlotName = Literal["llm", "embed", "ocr", "asr"]

class ModelRegistry:

    def __init__(self):
        self.slots: dict = {"llm": None, "embed": None, "ocr": None, "asr": None}
        self._restore()

    def _save(self):
        try:
            with open(_REGISTRY_PATH, "w") as f:
                json.dump(self.slots, f)
        except Exception as e:
            print(f"[registry] could not save state: {e}")

    def _restore(self):
        if not os.path.exists(_REGISTRY_PATH):
            return
        try:
            with open(_REGISTRY_PATH) as f:
                saved = json.load(f)
        except Exception as e:
            print(f"[registry] could not load saved state: {e}")
            return
        for slot, model_name in saved.items():
            if not model_name:
                continue
            print(f"[registry] restoring {slot} → {model_name}")
            try:
                if slot == "asr":
                    self._mount_asr(model_name)
                else:
                    self._mount_ollama(slot, model_name)
                self.slots[slot] = model_name
            except Exception as e:
                print(f"[registry] restore {slot}/{model_name} failed: {e}")

    def status(self) -> dict:
        return {
            "llm":   self.slots["llm"],
            "embed": self.slots["embed"],
            "ocr":   self.slots["ocr"],
            "asr":   self.slots["asr"],
        }

    def mount(self, slot: str, model_name: str):
        if slot == "asr":
            self._mount_asr(model_name)
        else:
            self._mount_ollama(slot, model_name)
        self.slots[slot] = model_name
        self._save()

    def unmount(self, slot: str):
        model = self.slots.get(slot)
        if not model:
            return
        if slot == "asr":
            audio_ingestor.unload_model()
        else:
            try:
                if slot == "embed":
                    ollama.embeddings(model=model, prompt="", keep_alive=0)
                else:
                    ollama.generate(model=model, prompt="", keep_alive=0)
            except Exception as e:
                print(f"[registry] unmount {slot}/{model} error (non-fatal): {e}")
        self.slots[slot] = None
        self._save()
        if slot == "llm":
            inference.llm_name = None

    def _mount_ollama(self, slot: str, model_name: str):
        available = inference.get_available_models()
        if model_name not in available:
            print(f"[registry] pulling {model_name}…")
            ollama.pull(model_name)
        try:
            if slot == "embed":
                ollama.embeddings(model=model_name, prompt="warmup", keep_alive=-1)
            else:
                ollama.generate(model=model_name, prompt="", keep_alive=-1)
        except Exception as e:
            print(f"[registry] warm-up {slot}/{model_name} error (non-fatal): {e}")
        if slot == "llm":
            inference.llm_name = model_name

    def _mount_asr(self, model_name: str):
        audio_ingestor.load_model(model_name)


registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LoadModelRequest(BaseModel):
    model_name: str

class DeleteModelRequest(BaseModel):
    model_name: str

class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 256
    temperature: float = 0.7

class QueryResponse(BaseModel):
    response: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[int] = None
    embedding_model: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"

class UpdateSessionRequest(BaseModel):
    title: str

class MountRequest(BaseModel):
    slot: str
    model_name: str

class UnmountRequest(BaseModel):
    slot: str

class WhisperModelRequest(BaseModel):
    model_name: str

class WhisperDeleteRequest(BaseModel):
    model_name: str


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "LLM Inference API is running"}

@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest):
    try:
        return db.create_session(title=request.title or "New Chat")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    try:
        return {"sessions": db.list_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: int):
    try:
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: int, request: UpdateSessionRequest):
    try:
        ok = db.update_session_title(session_id, request.title)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: int):
    try:
        ok = db.delete_session(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        delete_vector_store(session_id)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: int):
    try:
        return {"messages": db.get_messages(session_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/files")
async def get_session_files(session_id: int):
    try:
        return {"files": db.get_uploaded_files(session_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def get_models():
    return {
        "models": inference.get_available_models(),
        "categorized_models": inference.get_categorized_models(),
        "current_model": inference.llm_name,
    }

@app.get("/api/trending-models")
async def get_popular_models():
    try:
        response = requests.get("https://ollama.com/library?sort=popular", timeout=5)
        matches = re.findall(r'href="/library/([^/"]+)"', response.text)
        seen, popular = set(), []
        for match in matches:
            if match not in seen and match != "library":
                seen.add(match)
                popular.append(match)
                if len(popular) >= 8:
                    break
        if popular:
            return {"popular": popular}
    except Exception as e:
        print(f"Failed to fetch models from ollama.com: {e}")
    return {"popular": ["llama3.2", "mistral", "qwen2.5:0.5b", "gemma2", "phi3", "deepseek-coder-v2"]}

@app.post("/api/models/delete")
async def delete_model(request: DeleteModelRequest):
    try:
        inference.delete_model(request.model_name)
        return {"status": "success", "message": f"Model {request.model_name} deleted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/load")
async def load_model(request: LoadModelRequest):
    available_models = inference.get_available_models()
    if request.model_name not in available_models:
        def pull_stream():
            try:
                for progress in ollama.pull(request.model_name, stream=True):
                    d = vars(progress) if hasattr(progress, '__dict__') else progress if isinstance(progress, dict) else dict(progress)
                    yield json.dumps(d) + "\n"
                inference.llm_name = request.model_name
                yield json.dumps({"status": "success", "message": f"Model {request.model_name} loaded and ready."}) + "\n"
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
        return StreamingResponse(pull_stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})
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

@app.get("/api/models/status")
async def models_status():
    return registry.status()

@app.post("/api/models/mount")
async def mount_model(request: MountRequest):
    slot       = request.slot
    model_name = request.model_name

    if slot not in ("llm", "embed", "ocr", "asr"):
        return {"status": "error", "message": f"Unknown slot '{slot}'. Use llm, embed, ocr, or asr."}

    if slot == "asr":
        try:
            registry.mount("asr", model_name)
            return {"status": "success", "slot": "asr", "model": model_name}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def mount_stream():
        available = inference.get_available_models()
        if model_name not in available:
            try:
                for progress in ollama.pull(model_name, stream=True):
                    d = vars(progress) if hasattr(progress, '__dict__') else (
                        progress if isinstance(progress, dict) else dict(progress)
                    )
                    yield json.dumps(d) + "\n"
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)}) + "\n"
                return
        try:
            if slot == "embed":
                ollama.embeddings(model=model_name, prompt="warmup", keep_alive=-1)
            else:
                ollama.generate(model=model_name, prompt="", keep_alive=-1)
        except Exception as e:
            yield json.dumps({"status": "error", "message": f"Warm-up failed: {e}"}) + "\n"
            return
        registry.slots[slot] = model_name
        if slot == "llm":
            inference.llm_name = model_name
        registry._save()
        yield json.dumps({"status": "success", "slot": slot, "model": model_name}) + "\n"

    return StreamingResponse(mount_stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})

@app.post("/api/models/unmount")
async def unmount_model(request: UnmountRequest):
    if request.slot not in ("llm", "embed", "ocr", "asr"):
        return {"status": "error", "message": f"Unknown slot '{request.slot}'."}
    try:
        registry.unmount(request.slot)
        return {"status": "success", "slot": request.slot}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Generate / Chat
# ---------------------------------------------------------------------------

@app.post("/api/generate", response_model=QueryResponse)
async def generate_response(request: QueryRequest):
    response_text = inference.generate(
        prompt=request.query,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return {"response": response_text}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if inference.llm_name is None:
        return {"status": "error", "message": "No LLM model loaded. Please load a model first."}

    embedding_model = request.embedding_model or registry.slots.get("embed") or None

    history_lines = []
    user_query    = ""

    for msg in request.messages:
        if msg.role == "user":
            if msg.content.startswith("[File:"):
                lines  = msg.content.split("\n", 1)
                header = lines[0]
                body   = lines[1].strip() if len(lines) > 1 else ""
                fname  = header[len("[File: "):-1] if header.startswith("[File: ") and header.endswith("]") else header
                history_lines.append(f"Document context (Contents of {fname}):\n{body}" if body else f"Document context (Contents of {fname}): (empty)")
            else:
                if user_query:
                    history_lines.append(f"User: {user_query}")
                user_query = msg.content
        elif msg.role == "assistant":
            history_lines.append(f"Assistant: {msg.content}")

    system_instruction = (
        "You are a helpful assistant. "
        "When the conversation contains 'Document context:' blocks, use that "
        "information to answer the user's question. "
        "Do not say you cannot access files — the file contents are already "
        "provided as text in the conversation.\n\n"
    )

    has_inline_file_context = any(
        msg.role == "user" and msg.content.startswith("[File:")
        for msg in request.messages
    )

    context_block = ""
    vs = get_vector_store(request.session_id)

    if embedding_model and vs.index.ntotal > 0 and not has_inline_file_context:
        try:
            pipeline = EmbeddingPipeline(embedding_model=embedding_model)
            q_emb    = pipeline.embed_async(user_query).result()
            hits     = vs.search(np.array(q_emb, dtype=np.float32), k=5)
            hits     = [h for h in hits if h.get("score", 0) >= 0.3]

            if hits:
                    audio_hits    = [h for h in hits if h.get("category") == "audio"]
                    video_hits    = [h for h in hits if h.get("category") == "video_transcript"]
                    other_hits    = [h for h in hits if h.get("category") not in ("audio", "video_transcript")]
                    context_parts = []

                    if audio_hits:
                        audio_sources       = list(dict.fromkeys(h["source"] for h in audio_hits))
                        combined_transcript = "\n\n".join(h["text"] for h in audio_hits)
                        summary_prompt = (
                            f"The following are excerpts from the transcript of "
                            f"{', '.join(audio_sources)}.\n\n"
                            f"{combined_transcript}\n\n"
                            f"Based only on the above transcript excerpts, answer or summarise "
                            f"the following query in a clear, concise way:\n{user_query}"
                        )
                        audio_summary = inference.generate_async(prompt=summary_prompt).result()
                        context_parts.append(f"Summary from audio transcript ({', '.join(audio_sources)}):\n{audio_summary}")

                    if video_hits:
                        def _fmt_ts(seconds):
                            m, s = divmod(int(seconds), 60)
                            return f"{m:02d}:{s:02d}"

                        snippets = []
                        for h in video_hits:
                            start = h.get("start_time")
                            end   = h.get("end_time")
                            if start is not None and end is not None:
                                label = f"[{_fmt_ts(start)} – {_fmt_ts(end)}]"
                            elif start is not None:
                                label = f"[{_fmt_ts(start)}]"
                            else:
                                label = ""
                            snippets.append(f"{label} {h['text']}".strip())
                        context_parts.append(
                            f"Relevant video transcript excerpts ({h.get('source', 'video')}):\n"
                            + "\n\n".join(snippets)
                        )

                    if other_hits:
                        snippets = "\n\n".join(h["text"] for h in other_hits)
                        context_parts.append(f"Relevant context from your documents:\n{snippets}")

                    context_block = "\n\n".join(context_parts) + "\n\n"
        except Exception as e:
            print(f"RAG retrieval failed (non-fatal): {e}")

    history_block = "\n".join(history_lines)
    if history_block:
        history_block = f"Conversation so far:\n{history_block}\n\n"

    full_prompt = (
        f"{system_instruction}"
        f"{context_block}"
        f"{history_block}"
        f"User: {user_query}\n"
        f"Assistant:"
    )

    model      = inference.llm_name
    options    = {"num_predict": request.max_tokens, "temperature": request.temperature}
    session_id = request.session_id

    if session_id is not None:
        try:
            db.add_message(session_id, "user", user_query)
        except Exception as e:
            print(f"[db] failed to persist user message: {e}")

    def stream_tokens():
        accumulated = ""
        try:
            for chunk in ollama.generate(model=model, prompt=full_prompt, options=options, stream=True):
                token = chunk.get("response", "") if isinstance(chunk, dict) else getattr(chunk, "response", "")
                if token:
                    accumulated += token
                    yield json.dumps({"token": token}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            if session_id and accumulated:
                try:
                    db.add_message(session_id, "assistant", accumulated)
                    sess = db.get_session(session_id)
                    if sess and sess["title"] == "New Chat":
                        short_title = user_query[:60].strip()
                        if short_title:
                            db.update_session_title(session_id, short_title)
                except Exception as e:
                    print(f"[db] failed to persist assistant message: {e}")

    return StreamingResponse(stream_tokens(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# File ingest
# ---------------------------------------------------------------------------

@app.post("/api/files/ingest")
async def ingest_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
    embedding_model: Optional[str] = None,
    asr_model: Optional[str] = None,
):
    embedding_model = embedding_model or registry.slots.get("embed") or None
    asr_model       = asr_model       or registry.slots.get("asr")   or None

    sid: Optional[int] = None
    if session_id:
        try:
            sid = int(session_id)
        except (ValueError, TypeError):
            pass

    vs = get_vector_store(sid)

    results = []

    for upload in files:
        filename   = upload.filename or "unknown"
        ext        = os.path.splitext(filename)[1].lower()
        file_bytes = await upload.read()
        file_size  = len(file_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            if ext in AUDIO_EXTENSIONS:
                if audio_ingestor.model is None:
                    available = audio_ingestor.get_downloaded_models()
                    if available:
                        audio_ingestor.load_model(available[0])
                    else:
                        result = {"filename": filename, "category": "audio", "status": "error",
                                  "detail": "No Whisper model loaded. Load one first.", "chunks": 0, "text": ""}
                        results.append(result)
                        if sid:
                            try:
                                db.add_uploaded_file(sid, filename, "audio", "error", 0, result["detail"], file_size)
                            except Exception as dbe:
                                print(f"[db] file persist error: {dbe}")
                        continue

                transcribe_result = audio_ingestor.transcribe(tmp_path)
                text = str(transcribe_result.get("text", ""))

                chunks_created = 0
                summary        = ""

                if text.strip() and embedding_model:
                    if vs.has_source(filename):
                        existing = sum(1 for m in vs.metadata if m.get("source") == filename)
                        result = {"filename": filename, "category": "audio", "status": "duplicate",
                                  "detail": f"Already ingested ({existing} chunk(s) in vector store). Delete and re-upload to replace.",
                                  "chunks": existing, "text": text[:8000]}
                        results.append(result)
                        if sid:
                            try:
                                db.add_uploaded_file(sid, filename, "audio", "duplicate", existing, result["detail"][:500], file_size)
                            except Exception as dbe:
                                print(f"[db] file persist error: {dbe}")
                        continue

                    llm_future = inference.generate_async(prompt=f"Summarize this audio transcript in bullet points:\n{text}")
                    pipeline   = EmbeddingPipeline(embedding_model=embedding_model)
                    chunks, embed_futures = pipeline.process_document_async(text, filename, category="audio")
                    summary = llm_future.result()
                    if chunks:
                        embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)
                        vs.add_embeddings(embeddings, chunks)
                        chunks_created = len(chunks)

                result = {"filename": filename, "category": "audio", "status": "success",
                          "detail": summary or text[:200], "chunks": chunks_created, "text": text[:8000]}
                results.append(result)
                if sid:
                    try:
                        db.add_uploaded_file(sid, filename, "audio", "success", chunks_created, result["detail"][:500], file_size)
                        db.add_message(sid, "context", f"[File: {filename}]\n{text[:8000]}")
                    except Exception as dbe:
                        print(f"[db] file persist error: {dbe}")

            elif ext in VIDEO_EXTENSIONS:
                if audio_ingestor.model is None:
                    available_asr = audio_ingestor.get_downloaded_models()
                    if available_asr:
                        audio_ingestor.load_model(available_asr[0])

                transcript, transcript_segments = video_ingestor.process_video_with_audio(
                    tmp_path, filename, audio_ingestor
                )

                chunks_created = 0
                summary        = ""

                if vs.has_source(filename):
                    existing = sum(1 for m in vs.metadata if m.get("source") == filename)
                    result = {"filename": filename, "category": "video", "status": "duplicate",
                              "detail": f"Already ingested ({existing} chunk(s) in vector store). Delete and re-upload to replace.",
                              "chunks": existing, "text": transcript[:8000]}
                    results.append(result)
                    if sid:
                        try:
                            db.add_uploaded_file(sid, filename, "video", "duplicate", existing, result["detail"][:500], file_size)
                        except Exception as dbe:
                            print(f"[db] file persist error: {dbe}")
                    continue

                if transcript.strip() and embedding_model:
                    llm_future = None
                    if inference.llm_name is not None:
                        llm_future = inference.generate_async(
                            prompt=f"Summarize this video transcript in bullet points:\n{transcript}",
                            max_tokens=1024,
                        )
                    pipeline = EmbeddingPipeline(embedding_model=embedding_model)
                    if transcript_segments:
                        txt_chunks = pipeline.chunk_segments(
                            transcript_segments, source=filename, category="video_transcript"
                        )
                        embed_futures = [pipeline.embed_async(c["text"]) for c in txt_chunks]
                    else:
                        txt_chunks, embed_futures = pipeline.process_document_async(
                            transcript, filename, category="video_transcript"
                        )
                    if llm_future is not None:
                        summary = llm_future.result()
                    if txt_chunks:
                        embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)
                        vs.add_embeddings(embeddings, txt_chunks)
                        chunks_created += len(txt_chunks)

                result = {"filename": filename, "category": "video", "status": "success",
                          "detail": summary or transcript[:200] or "Processed video — no audio transcript available.",
                          "chunks": chunks_created, "text": transcript[:8000]}
                results.append(result)
                if sid:
                    try:
                        db.add_uploaded_file(sid, filename, "video", "success", chunks_created, result["detail"][:500], file_size)
                        if transcript.strip():
                            db.add_message(sid, "context", f"[File: {filename}]\n{transcript[:8000]}")
                    except Exception as dbe:
                        print(f"[db] file persist error: {dbe}")

            elif ext in IMAGE_EXTENSIONS:
                active_ocr_model = registry.slots.get("ocr") or ""
                text = doc_ingestor.ocr_async(tmp_path, ocr_model=active_ocr_model).result()

                chunks_created = 0

                if text.strip() and embedding_model:
                    if vs.has_source(filename):
                        existing = sum(1 for m in vs.metadata if m.get("source") == filename)
                        result = {"filename": filename, "category": "image", "status": "duplicate",
                                  "detail": f"Already ingested ({existing} chunk(s) in vector store). Delete and re-upload to replace.",
                                  "chunks": existing, "text": text[:8000]}
                        results.append(result)
                        if sid:
                            try:
                                db.add_uploaded_file(sid, filename, "image", "duplicate", existing, result["detail"][:500], file_size)
                            except Exception as dbe:
                                print(f"[db] file persist error: {dbe}")
                        continue

                    pipeline = EmbeddingPipeline(embedding_model=embedding_model)
                    chunks, embed_futures = pipeline.process_document_async(text, filename)
                    if chunks:
                        embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)
                        vs.add_embeddings(embeddings, chunks)
                        chunks_created = len(chunks)

                ocr_status = "success" if text.strip() else "error"
                ocr_detail = (
                    text[:200] if text.strip()
                    else ("No OCR model mounted — mount a vision model in the OCR slot first"
                          if not active_ocr_model else "OCR model returned no text")
                )
                result = {"filename": filename, "category": "image", "status": ocr_status,
                          "detail": ocr_detail, "chunks": chunks_created,
                          "text": text[:8000] if text.strip() else ""}
                results.append(result)
                if sid:
                    try:
                        db.add_uploaded_file(sid, filename, "image", ocr_status, chunks_created, ocr_detail[:500], file_size)
                        if ocr_status == "success" and text.strip():
                            db.add_message(sid, "context", f"[File: {filename}]\n{text[:8000]}")
                    except Exception as dbe:
                        print(f"[db] file persist error: {dbe}")

            elif ext in TEXT_EXTENSIONS:
                with open(tmp_path, "r", errors="replace") as f:
                    text = f.read()

                chunks_created = 0

                if text.strip() and embedding_model:
                    if vs.has_source(filename):
                        existing = sum(1 for m in vs.metadata if m.get("source") == filename)
                        result = {"filename": filename, "category": "text", "status": "duplicate",
                                  "detail": f"Already ingested ({existing} chunk(s) in vector store). Delete and re-upload to replace.",
                                  "chunks": existing, "text": text[:8000]}
                        results.append(result)
                        if sid:
                            try:
                                db.add_uploaded_file(sid, filename, "text", "duplicate", existing, result["detail"][:500], file_size)
                            except Exception as dbe:
                                print(f"[db] file persist error: {dbe}")
                        continue

                    pipeline = EmbeddingPipeline(embedding_model=embedding_model)
                    chunks, embed_futures = pipeline.process_document_async(text, filename)
                    if chunks:
                        embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)
                        vs.add_embeddings(embeddings, chunks)
                        chunks_created = len(chunks)

                txt_status = "success" if text.strip() else "error"
                txt_detail = text[:200] if text.strip() else "File is empty"
                result = {"filename": filename, "category": "text", "status": txt_status,
                          "detail": txt_detail, "chunks": chunks_created,
                          "text": text[:8000] if text.strip() else ""}
                results.append(result)
                if sid:
                    try:
                        db.add_uploaded_file(sid, filename, "text", txt_status, chunks_created, txt_detail[:500], file_size)
                        if txt_status == "success":
                            db.add_message(sid, "context", f"[File: {filename}]\n{text[:8000]}")
                    except Exception as dbe:
                        print(f"[db] file persist error: {dbe}")

            elif ext in PDF_EXTENSIONS:
                pdf_texts        = []
                active_ocr_model = registry.slots.get("ocr") or ""

                try:
                    with pdfplumber.open(tmp_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                pdf_texts.append(page_text)
                            elif active_ocr_model:
                                try:
                                    pil_img = page.to_image(resolution=150).original
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as ptmp:
                                        pil_img.save(ptmp.name, format="PNG")
                                        page_ocr = doc_ingestor.ocr_async(ptmp.name, ocr_model=active_ocr_model).result()
                                    os.unlink(ptmp.name)
                                    if page_ocr.strip():
                                        pdf_texts.append(page_ocr)
                                except Exception as pe:
                                    print(f"[PDF] page OCR failed: {pe}")
                except Exception as e:
                    result = {"filename": filename, "category": "pdf", "status": "error",
                              "detail": f"Could not read PDF: {e}", "chunks": 0, "text": ""}
                    results.append(result)
                    if sid:
                        try:
                            db.add_uploaded_file(sid, filename, "pdf", "error", 0, result["detail"][:500], file_size)
                        except Exception as dbe:
                            print(f"[db] file persist error: {dbe}")
                    continue

                text           = "\n\n".join(pdf_texts)
                chunks_created = 0

                if text.strip() and embedding_model:
                    if vs.has_source(filename):
                        existing = sum(1 for m in vs.metadata if m.get("source") == filename)
                        result = {"filename": filename, "category": "pdf", "status": "duplicate",
                                  "detail": f"Already ingested ({existing} chunk(s) in vector store). Delete and re-upload to replace.",
                                  "chunks": existing, "text": text[:8000]}
                        results.append(result)
                        if sid:
                            try:
                                db.add_uploaded_file(sid, filename, "pdf", "duplicate", existing, result["detail"][:500], file_size)
                            except Exception as dbe:
                                print(f"[db] file persist error: {dbe}")
                        continue

                    pipeline = EmbeddingPipeline(embedding_model=embedding_model)
                    chunks, embed_futures = pipeline.process_document_async(text, filename)
                    if chunks:
                        embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)
                        vs.add_embeddings(embeddings, chunks)
                        chunks_created = len(chunks)

                pdf_status = "success" if text.strip() else "error"
                pdf_detail = text[:200] if text.strip() else "No text extracted from PDF"
                result = {"filename": filename, "category": "pdf", "status": pdf_status,
                          "detail": pdf_detail, "chunks": chunks_created,
                          "text": text[:8000] if text.strip() else ""}
                results.append(result)
                if sid:
                    try:
                        db.add_uploaded_file(sid, filename, "pdf", pdf_status, chunks_created, pdf_detail[:500], file_size)
                        if pdf_status == "success":
                            db.add_message(sid, "context", f"[File: {filename}]\n{text[:8000]}")
                    except Exception as dbe:
                        print(f"[db] file persist error: {dbe}")

            else:
                result = {"filename": filename, "category": "unknown", "status": "error",
                          "detail": f"Unsupported file type: {ext}", "chunks": 0, "text": ""}
                results.append(result)
                if sid:
                    try:
                        db.add_uploaded_file(sid, filename, "unknown", "error", 0, result["detail"][:500], file_size)
                    except Exception as dbe:
                        print(f"[db] file persist error: {dbe}")

        except Exception as e:
            result = {"filename": filename, "category": "unknown", "status": "error",
                      "detail": str(e), "chunks": 0, "text": ""}
            results.append(result)
            if sid:
                try:
                    db.add_uploaded_file(sid, filename, "unknown", "error", 0, str(e)[:500], file_size)
                except Exception as dbe:
                    print(f"[db] file persist error: {dbe}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return {"results": results}


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

@app.get("/api/whisper/models")
async def get_whisper_models():
    return {
        "models": audio_ingestor.get_downloaded_models(),
        "all_models": audio_ingestor.get_available_models(),
        "current_model": audio_ingestor.model_name,
    }

@app.post("/api/whisper/models/load")
async def load_whisper_model(request: WhisperModelRequest):
    if request.model_name not in audio_ingestor.get_available_models():
        return {"status": "error", "message": f"Model {request.model_name} not available."}

    if audio_ingestor.model_name == request.model_name and audio_ingestor.model is not None:
        return {"status": "success", "message": f"Whisper model {request.model_name} already loaded."}

    model_sizes  = {"tiny": 75, "base": 75, "small": 150, "medium": 300, "large": 400, "large-v2": 400, "large-v3": 400}
    total_steps  = model_sizes.get(request.model_name, 200)

    def generate_progress():
        try:
            for step in range(1, total_steps + 1):
                progress = int((step / total_steps) * 100)
                yield json.dumps({"status": "downloading", "completed": progress, "total": 100,
                                  "message": f"Downloading Whisper model {request.model_name}..."}) + "\n"
                if step % 20 == 0:
                    time.sleep(0.1)
            audio_ingestor.load_model(request.model_name)
            yield json.dumps({"status": "success", "completed": 100, "total": 100,
                              "message": f"Whisper model {request.model_name} loaded successfully."}) + "\n"
        except Exception as e:
            time.sleep(0.1)
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(generate_progress(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})

@app.post("/api/whisper/models/unload")
async def unload_whisper_model():
    try:
        audio_ingestor.unload_model()
        return {"status": "success", "message": "Whisper model unloaded successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/whisper/models/delete")
async def delete_whisper_model(request: WhisperDeleteRequest):
    try:
        if audio_ingestor.model_name == request.model_name:
            audio_ingestor.unload_model()
        cache_dir  = os.path.expanduser("~/.cache/whisper")
        model_path = os.path.join(cache_dir, f"{request.model_name}.pt")
        if os.path.exists(model_path):
            os.remove(model_path)
            return {"status": "success", "message": f"Whisper model {request.model_name} deleted from cache."}
        return {"status": "success", "message": f"Whisper model {request.model_name} not found in cache (may already be deleted)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Legacy endpoints
# ---------------------------------------------------------------------------

@app.post("/api/audio/digest")
async def audio_digest(file: UploadFile = File(...), embedding_model: str = ...):
    if not file.filename:
        return {"status": "error", "message": "No filename provided"}

    filename      = file.filename
    file_location = f"temp_{filename}"

    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        transcribe_result = audio_ingestor.transcribe(file_location)
        text = str(transcribe_result["text"])

        llm_future       = inference.generate_async(prompt=f"Summarize this file in bullet points\n{text}")
        embedding_pipeline = EmbeddingPipeline(embedding_model=embedding_model)
        chunks, embed_futures = embedding_pipeline.process_document_async(text, filename)

        summary    = llm_future.result()
        embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)

        if len(chunks) > 0:
            get_vector_store(None).add_embeddings(embeddings, chunks)

        return {"transcribe": text, "summary": summary}
    finally:
        os.remove(file_location)

@app.post("/api/documents/ingest")
async def ingest_document(file: UploadFile = File(...), embedding_model: str = ...):
    try:
        if not file.filename:
            return {"status": "error", "message": "No filename provided"}

        filename           = file.filename
        embedding_pipeline = EmbeddingPipeline(embedding_model=embedding_model)

        with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                active_ocr_model = registry.slots.get("ocr") or ""
                text = doc_ingestor.ocr_async(tmp_path, ocr_model=active_ocr_model).result()
                if not text.strip():
                    return {"status": "error", "message": "No text extracted from document"}
                chunks, embed_futures = embedding_pipeline.process_document_async(text, filename)
                if not chunks:
                    return {"status": "error", "message": "No chunks created from document"}
                embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)

            elif filename.lower().endswith(('.txt', '.md')):
                with open(tmp_path, 'r') as f:
                    text = f.read()
                if not text.strip():
                    return {"status": "error", "message": "No text extracted from document"}
                chunks, embed_futures = embedding_pipeline.process_document_async(text, filename)
                if not chunks:
                    return {"status": "error", "message": "No chunks created from document"}
                embeddings = EmbeddingPipeline.collect_embeddings(embed_futures)

            else:
                return {"status": "error", "message": f"Unsupported file type: {filename}"}

            get_vector_store(None).add_embeddings(embeddings, chunks)
            return {
                "status": "success",
                "message": f"Document '{filename}' ingested successfully",
                "chunks_created": len(chunks),
                "doc_id": chunks[0]['doc_id'] if chunks else None,
            }
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return {"status": "error", "message": str(e)}
