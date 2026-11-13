import io
from fastapi import FastAPI,HTTPException,UploadFile
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI()

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(status_code=400, detail='Upload a valid image')

    return{
        "filename": file.filename,
        "label": "ai_generated",
        "probability": 0.75,
        "is_placeholder": True,
    }



