import os
import io
import torch
import huggingface_hub
import asyncio
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import StreamingResponse
from PIL import Image
import uvicorn

if not hasattr(torch, 'xpu'):
    class XPU:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def empty_cache(): pass
    torch.xpu = XPU

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

os.environ["DIFFUSERS_VERIFY_COMPATIBILITY"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from pipeline import reconstruct_artifact

app = FastAPI(title="AI Artifact Reconstruction")

@app.get("/")
async def home():
    return {
        "status": "online",
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__
    }

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()

    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        Image.open(io.BytesIO(content)).verify()
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        processed_img = await asyncio.wait_for(
            asyncio.to_thread(reconstruct_artifact, content),
            timeout=60
        )

        img_byte_arr = io.BytesIO()
        processed_img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
