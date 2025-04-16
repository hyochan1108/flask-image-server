from flask import Flask, request, jsonify
from ultralytics import YOLO
import requests, os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] YOLO 모델 다운로드 중...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("[INFO] YOLO 모델 다운로드 완료")
    return YOLO(MODEL_PATH)

@app.route('/upload', methods=['POST'])
def upload_image():
    image = request.files['image']
    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(save_path)

    try:
        model = load_model()
        results = model(save_path)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'message': f'✅ 분석 완료: {image.filename}',
        'boxes': boxes,
        'classes': classes
    })
