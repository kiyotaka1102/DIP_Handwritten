import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
from io import BytesIO
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import easyocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import logging
from transformers import pipeline
from pyvi import ViUtils

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI(title="Vietnamese Text Refinement API")

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
    global reader, detector, phobert_pipeline
    reader = easyocr.Reader(['vi'])
    
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained'] = True
    config['device'] = 'cpu'
    config['weights'] = 'D:/DigitalImageProcessing/Final/DIP_Handwritten/src/weights/transformerocr.pth'
    
    try:
        detector = Predictor(config)
        logging.info("VietOCR initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize VietOCR: {str(e)}")
        raise

    try:
        phobert_pipeline = pipeline(
            "fill-mask",
            model="vinai/phobert-base",
            tokenizer="vinai/phobert-base",
            device=-1
        )
        logging.info("PhoBERT pipeline initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize PhoBERT: {str(e)}")
        raise

def is_valid_vietnamese(text):
    try:
        normalized = ViUtils.add_accents(text)
        return len(normalized.errors) == 0
    except Exception:
        return False

def refine_text_with_phobert(text, threshold=0.3):
    try:
        if not text.strip():
            return text

        masked_text = f"{text} <mask>"
        predictions = phobert_pipeline(masked_text, top_k=5)
        
        for pred in predictions:
            candidate = pred["sequence"].replace("<s>", "").replace("</s>", "").strip()
            if is_valid_vietnamese(candidate):
                return candidate

        words = text.split()
        refined_words = []
        for i, word in enumerate(words):
            if is_valid_vietnamese(word):
                refined_words.append(word)
                continue
            
            masked_word = " ".join([w if j != i else "<mask>" for j, w in enumerate(words)])
            predictions = phobert_pipeline(masked_word, top_k=3)
            for pred in predictions:
                candidate = pred["token_str"]
                if is_valid_vietnamese(candidate):
                    refined_words.append(candidate)
                    break
            else:
                refined_words.append(word)
                
        return " ".join(refined_words)
        
    except Exception as e:
        logging.error(f"Refinement error: {str(e)}")
        return text

def preprocess_image(image):
    img = image.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    return img.convert('RGB')

def detect_text_regions(image, detect_paragraph=True, min_confidence=0.5):
    try:
        results = reader.readtext(
            np.array(image),
            decoder='wordbeamsearch',
            min_size=20,
            text_threshold=0.4,
            link_threshold=0.4,
            width_ths=0.7,
            paragraph=detect_paragraph
        )
        
        boxes = []
        for detection in results:
            if len(detection) == 3:
                bbox, text, conf = detection
                if conf < min_confidence:
                    continue  # skip low-confidence detections
            else:
                bbox, text = detection
                conf = None  # fallback if confidence not returned
            
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            boxes.append((
                int(min(x_coords)), int(min(y_coords)),
                int(max(x_coords)), int(max(y_coords))
            ))
        return boxes

    except Exception as e:
        logging.error(f"Error during text region detection: {str(e)}")
        return []

def process_image(image, use_scene_text_detection=False, refine=True, detect_paragraph=True):
    enhanced_img = preprocess_image(image)
    img_with_boxes = enhanced_img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    detected_texts = []
    print(detect_paragraph,refine, use_scene_text_detection )
    if use_scene_text_detection:
        boxes = detect_text_regions(enhanced_img,detect_paragraph, min_confidence= 0.1)

        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            try:
                cropped_img = enhanced_img.crop((xmin, ymin, xmax, ymax))
                prediction = detector.predict(cropped_img)
                if refine:
                    prediction = refine_text_with_phobert(prediction)

                final_text = ViUtils.remove_accents(prediction).decode('utf-8') if is_valid_vietnamese(prediction) else prediction

                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                draw.text((xmin, ymin - 10), final_text, fill="red")

                buffered = BytesIO()
                cropped_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                detected_texts.append({
                    "id": i + 1,
                    "text": final_text,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "image": img_str
                })
            except Exception as e:
                logging.error(f"Error processing region {i}: {str(e)}")
                continue
    else:
        try:
            prediction = detector.predict(enhanced_img)
            if refine:
                prediction = refine_text_with_phobert(prediction)

            final_text = ViUtils.remove_accents(prediction).decode('utf-8') if is_valid_vietnamese(prediction) else prediction

            detected_texts.append({
                "id": 1,
                "text": final_text,
                "bbox": None,
                "image": None
            })

        except Exception as e:
            logging.error(f"Error processing full image with VietOCR: {str(e)}")

    def image_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    return {
        "original_image": image_to_base64(image),
        "processed_image": image_to_base64(img_with_boxes),
        "detected_texts": detected_texts,
        "count": len(detected_texts)
    }

@app.get("/", response_class=HTMLResponse)
async def get_html():
    try:
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
async def detect_text(
    file: UploadFile = File(...),
    use_scene_text_detection: bool = Query(True),
    refine: bool = Form(True),
    detect_paragraph: bool = Query(True)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, detail="Invalid image format. Please upload a JPEG or PNG file.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        result = process_image(image, use_scene_text_detection, refine, detect_paragraph)
        return result
    except Exception as e:
        import traceback
        logging.error(f"Error during /detect/: {traceback.format_exc()}")
        raise HTTPException(500, detail=f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
