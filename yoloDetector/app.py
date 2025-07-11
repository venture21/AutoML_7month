from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO('yolov8n.pt')  # 사전 훈련된 YOLOv8 모델 사용

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미합니다. 파일로 하려면 파일 경로를 입력하세요.

# 자동차 클래스 ID (COCO 데이터셋 기준)
CAR_CLASS_ID = 2

# 누적 자동차 대수
total_car_count = 0
detected_ids = set()

def generate_frames():
    global total_car_count
    global detected_ids

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # YOLO 모델을 사용하여 객체 탐지 및 추적
            results = model.track(frame, persist=True, classes=[CAR_CLASS_ID])

            # 결과 프레임 가져오기
            annotated_frame = results[0].plot()

            # 탐지된 자동차 수 계산
            if results[0].boxes.id is not None:
                current_ids = set(results[0].boxes.id.int().cpu().tolist())
                newly_detected_ids = current_ids - detected_ids
                total_car_count += len(newly_detected_ids)
                detected_ids.update(newly_detected_ids)


            # 화면에 누적 자동차 대수 표시
            cv2.putText(annotated_frame, f'Total Cars: {total_car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
