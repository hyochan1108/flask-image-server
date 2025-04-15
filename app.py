from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLO 모델 로드 (yolov8n.pt 또는 학습된 best.pt)
model = YOLO('yolov8n.pt')  # 모델 경로를 'best.pt'로 바꿔도 됨

@app.route('/')
def home():
    return '✅ YOLO 이미지 분석 서버 작동 중입니다.'

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': '파일 이름이 없습니다.'}), 400

    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(save_path)

    try:
        results = model(save_path)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()
    except Exception as e:
        return jsonify({
            'error': 'YOLO 분석 중 오류 발생',
            'detail': str(e)
        }), 500

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
