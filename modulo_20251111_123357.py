# módulo criado autonomamente pela Lúmen
from fastapi import FastAPI, UploadFile
from PIL import Image
import io

app = FastAPI()

@app.post("/upload")
async def upload_image(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read()))
    print(f"[INFO] Imagem recebida: {img.size}")
    return {"status": "ok", "width": img.width, "height": img.height}
