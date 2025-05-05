import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
from io import BytesIO
import uuid
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
import numpy as np
import easyocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch
import logging
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI(title="Scene Text Detection API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directory
os.makedirs("temp", exist_ok=True)

@app.on_event("startup")
async def startup_event():
    global reader, detector, summarizer
    reader = easyocr.Reader(['en', 'vi'])
    
    # Load VietOCR configuration
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['device'] = 'cpu'
    config['weights'] = 'D:/DigitalImageProcessing/Final/DIP_Handwritten/src/weights/vgg_transformer.pth'
    
    try:
        detector = Predictor(config)
        logging.info("VietOCR initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize VietOCR: {str(e)}")
        raise
    
    # Initialize the summarization model
    try:
        summarizer = pipeline("summarization", model="t5-small", device=-1)  # CPU
        logging.info("Summarizer initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize summarizer: {str(e)}")
        raise

def detect_text_regions(image):
    results = reader.readtext(np.array(image))
    boxes = []
    for detection in results:
        bbox, text, confidence = detection
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

def process_image(image, use_scene_text_detection=True):
    global summarizer
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    detected_texts = []
    
    if use_scene_text_detection:
        boxes = detect_text_regions(image)
        
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            cropped_img = image.crop((xmin, ymin, xmax, ymax))
            prediction = detector.predict(cropped_img)
            
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            draw.text((xmin, ymin - 10), prediction, fill="red")
            
            buffered = BytesIO()
            cropped_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            detected_texts.append({
                "id": i + 1,
                "text": prediction,
                "bbox": [xmin, ymin, xmax, ymax],
                "image": img_str
            })
    
    # Generate a summarized sentence
    text_to_summarize = " ".join([item["text"] for item in detected_texts])
    summarized_sentence = (
        summarizer(text_to_summarize, max_length=30, min_length=5, do_sample=False)[0]["summary_text"]
        if text_to_summarize else "No text detected in the image."
    )
    
    # Convert images to base64
    buffered = BytesIO()
    img_with_boxes.save(buffered, format="JPEG")
    img_with_boxes_str = base64.b64encode(buffered.getvalue()).decode()
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    original_img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "original_image": original_img_str,
        "processed_image": img_with_boxes_str,
        "detected_texts": detected_texts,
        "count": len(detected_texts),
        "summarized_sentence": summarized_sentence
    }

@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
        # Use path to static directory for index.html
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        html_path = os.path.join(base_dir, "static", "index.html")
        logging.info(f"Attempting to open: {html_path}")
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail="index.html not found")
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/detect/")  
async def detect_text(file: UploadFile = File(...), use_scene_text_detection: bool = Query(True, description="Whether to use scene text detection")):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, detail="Invalid image format. Please upload a JPEG or PNG file.")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        result = process_image(image, use_scene_text_detection)
        return result
    except Exception as e:
        raise HTTPException(500, detail=f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)