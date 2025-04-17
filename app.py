from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLO 모델 로딩
model = YOLO("yolov8n.pt")  # 이미 같은 폴더에 있음

@app.route('/')
def index():
    return "✅ YOLO 이미지 분석 서버 실행 중"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': '파일 이름이 없습니다'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(save_path)

    try:
        results = model(save_path)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
        return jsonify({
            'message': f'✅ 이미지 저장 및 분석 완료: {image.filename}',
            'boxes': boxes,
            'classes': classes
        }), 200
    except Exception as e:
        return jsonify({'error': 'YOLO 분석 실패', 'detail': str(e)}), 500

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render가 제공하는 포트번호 받아오기
    print(f"✅ 서버 시작: 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)

