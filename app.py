from flask import Flask, request, jsonify, send_from_directory
import os
import requests
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔽 YOLO 모델 다운로드 (최초 1회만 실행됨)
model_path = 'yolov8n.pt'
if not os.path.exists(model_path):
    print("[INFO] YOLO 모델 다운로드 중...")
    url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'
    r = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(r.content)
    print("[INFO] YOLO 모델 다운로드 완료")

# 🔽 모델 불러오기
model = YOLO(model_path)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(save_path)

    try:
        results = model(save_path)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
    except Exception as e:
        return jsonify({'error': 'YOLO 분석 실패', 'detail': str(e)}), 500

    return jsonify({
        'message': f'✅ 이미지 저장 및 분석 완료: {image.filename}',
        'boxes': boxes,
        'classes': classes
    }), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
