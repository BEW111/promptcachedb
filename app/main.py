import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

SERVER_CACHE_DIR = "./server_prompt_cache"
os.makedirs(SERVER_CACHE_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "promptcachedb server"}


@app.post("/upload/")
async def upload_safetensor(prompt_cache_file: UploadFile = File(...)):
    """Upload a safetensors file to the server"""
    if not prompt_cache_file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    
    file_path = os.path.join(SERVER_CACHE_DIR, prompt_cache_file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(prompt_cache_file.file, buffer)
        return {"message": f"File '{prompt_cache_file.filename}' uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.get("/load/{prompt_cache_file_id}")
async def load_safetensor(prompt_cache_file_id: str):
    """Stream a safetensors file"""
    file_path = os.path.join(SERVER_CACHE_DIR, f"{prompt_cache_file_id}.safetensors")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    def iterfile():
        with open(file_path, mode="rb") as file_like:
            yield from file_like

    # def iterfile():
    #     with open(file_path, mode="rb") as file_like:
    #         CHUNK_SIZE = 1024 * 1024
    #         while chunk := file_like.read(CHUNK_SIZE):
    #             yield chunk
    
    return StreamingResponse(iterfile(), media_type="application/octet-stream", headers={
        'Content-Disposition': f'attachment; filename="{prompt_cache_file_id}.safetensors"'
    })


@app.post("/clear_cache/")
async def clear_cache():
    """Remove all files from the cache on this server"""
    folder = Path(SERVER_CACHE_DIR)
    for file in folder.glob("*.safetensors"):
        try:
            file.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete {file}. Reason: {e}")

    return {"message": "Cleaned cache succesfully!"}