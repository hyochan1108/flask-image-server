from flask import Flask, request, jsonify, send_from_directory
import os
from ultralytics import YOLO
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_URL = 'https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt'
MODEL_PATH = 'yolov8n.pt'

# ✅ YOLO 모델 다운로드 (처음 실행 시)
if not os.path.exists(MODEL_PATH):
    try:
        print("[INFO] YOLO 모델 다운로드 시작...")
        r = requests.get(MODEL_URL)
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print("[INFO] YOLO 모델 다운로드 완료")
    except Exception as e:
        print(f"[ERROR] YOLO 모델 다운로드 실패: {e}")

# ✅ YOLO 모델 로딩
try:
    model = YOLO(MODEL_PATH)
    print("[INFO] YOLO 모델 로딩 완료")
except Exception as e:
    print(f"[ERROR] YOLO 모델 로딩 실패: {e}")
    model = None


@app.route('/')
def index():
    return '✅ 이미지 수신 서버 실행 중입니다.'


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    image = request.files['image']
    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(save_path)

    # YOLO 분석
    try:
        if model is None:
            raise Exception("YOLO 모델이 로딩되지 않았습니다.")

        results = model(save_path)
        if not results or not hasattr(results[0], "boxes"):
            raise ValueError("YOLO 결과가 없습니다.")

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

