from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os
from processor import process_video

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())

    input_path = f"{UPLOAD_DIR}/{job_id}.mp4"
    output_path = f"{OUTPUT_DIR}/{job_id}_silent.mp4"
    final_path = f"{OUTPUT_DIR}/{job_id}_final.mp4"

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process video (your code)
    process_video(input_path, output_path, final_path)

    return {
        "job_id": job_id,
        "output_video": final_path
    }