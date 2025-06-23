# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import cv2
import os
import datetime
import logging
import time
import pytesseract
import re
from PIL import Image
import numpy as np

app = Flask(__name__)

# Cấu hình logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Camera RTSP (hoặc webcam để test)
camera_url = os.getenv('CAMERA_URL', 'rtsp://admin:cctv123456@192.168.88.246/Streaming/Channels/101')
# Để test với webcam, uncomment dòng dưới và comment dòng trên
#camera_url = 2
cap = None

def init_camera():
    global cap
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            logging.info(f"Thử kết nối camera (lần {attempt}/{max_attempts}) với URL: {camera_url}")
            if cap is not None:
                cap.release()
                cap = None
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)
            cap.set(cv2.CAP_PROP_FPS, 10)
            time.sleep(2)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    logging.info("Kết nối camera thành công và đọc được frame")
                    return True
                else:
                    logging.warning("Mở camera nhưng không đọc được frame")
            else:
                logging.warning("Không thể mở camera")
            cap.release()
            cap = None
        except Exception as e:
            logging.error(f"Lỗi khởi tạo camera: {str(e)}")
        time.sleep(3)
    logging.error("Không thể kết nối camera sau nhiều lần thử")
    return False

@app.route('/')
def index():
    logging.info("Truy cập trang chủ")
    if not cap or not cap.isOpened():
        init_camera()
    return render_template('index.html')

@app.route('/video')
def video():
    def generate_frames():
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Camera Feed", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while True:
            try:
                if not cap or not cap.isOpened():
                    logging.info("Camera không sẵn sàng cho video, thử khởi tạo lại")
                    init_camera()
                success, frame = cap.read()
                if not success or frame is None:
                    logging.warning("Không đọc được frame, sử dụng placeholder")
                    frame = placeholder
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                logging.error(f"Lỗi khi stream video: {str(e)}")
                init_camera()
                frame = placeholder
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.5)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global cap
    try:
        logging.info("Nhận yêu cầu chụp ảnh")
        if not cap or not cap.isOpened():
            logging.warning("Camera chưa sẵn sàng, thử khởi tạo lại")
            if not init_camera():
                logging.error("Không thể kết nối camera")
                return jsonify({"error": "Không thể kết nối camera"}), 500
        ret, frame = cap.read()
        logging.info(f"Đọc frame: success={ret}, frame_shape={frame.shape if frame is not None else 'None'}")
        if not ret or frame is None:
            logging.error("Không đọc được frame từ camera")
            cap.release()
            cap = None
            init_camera()
            return jsonify({"error": "Không đọc được frame từ camera"}), 500
        
        now = datetime.datetime.now()
        date_folder = now.strftime('%Y-%m-%d')
        save_dir = os.path.join('captured', date_folder)
        logging.info(f"Tạo thư mục: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = now.strftime('%Y%m%d_%H%M%S')
        filename = f'captured_{timestamp}.jpg'
        filepath = os.path.join(save_dir, filename)
        logging.info(f"Thử lưu ảnh tại: {filepath}")
        if not cv2.imwrite(filepath, frame):
            logging.error(f"Không thể lưu ảnh vào {filepath}")
            return jsonify({"error": "Không thể lưu ảnh"}), 500
        logging.info(f"Đã lưu ảnh tại {filepath}")

        ocr_result = perform_ocr(filepath)
        logging.info(f"Kết quả OCR: {ocr_result}")
        return jsonify({"message": f"Đã lưu ảnh: {filename}", "ocr": ocr_result}), 200
    except Exception as e:
        logging.error(f"Lỗi khi chụp ảnh: {str(e)}")
        if cap is not None:
            cap.release()
            cap = None
        init_camera()
        return jsonify({"error": f"Lỗi khi chụp ảnh: {str(e)}"}), 500

def perform_ocr(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None:
            logging.error("Không thể đọc file ảnh để OCR")
            return {"error": "Không thể đọc file ảnh"}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_pil = Image.fromarray(thresh)

        text = pytesseract.image_to_string(img_pil, lang='vie', config='--psm 6')
        logging.info(f"Văn bản OCR thô: {text}")
        data = extract_data(text)
        return data
    except Exception as e:
        logging.error(f"Lỗi OCR: {str(e)}")
        return {"error": str(e)}

def extract_data(text):
    data = {"text": text.strip()}
    return data

@app.route('/gallery')
def gallery():
    try:
        logging.info("Truy cập trang gallery")
        date = request.args.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
        folder = os.path.join('captured', date)
        images = []
        if os.path.exists(folder):
            try:
                images = sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])
                logging.info(f"Tìm thấy {len(images)} ảnh trong thư mục {folder}")
            except Exception as e:
                logging.error(f"Lỗi khi liệt kê ảnh trong {folder}: {str(e)}")
                images = []
        else:
            logging.warning(f"Thư mục {folder} không tồn tại")

        available_dates = []
        try:
            available_dates = sorted(
                [d for d in os.listdir('captured') if os.path.isdir(os.path.join('captured', d))],
                reverse=True
            )
            logging.info(f"Các ngày khả dụng: {available_dates}")
        except Exception as e:
            logging.error(f"Lỗi khi liệt kê thư mục captured: {str(e)}")
            available_dates = []

        return render_template('gallery.html', date=date, images=images, available_dates=available_dates)
    except Exception as e:
        logging.error(f"Lỗi trong endpoint gallery: {str(e)}")
        return jsonify({"error": f"Lỗi khi tải gallery: {str(e)}"}), 500

@app.route('/image/<date>/<filename>')
def get_image(date, filename):
    try:
        folder = os.path.join('captured', date)
        if not os.path.exists(os.path.join(folder, filename)):
            logging.warning(f"Ảnh không tồn tại: {os.path.join(folder, filename)}")
            return jsonify({"error": "Ảnh không tồn tại"}), 404
        logging.info(f"Phục vụ ảnh: {os.path.join(folder, filename)}")
        return send_from_directory(folder, filename)
    except Exception as e:
        logging.error(f"Lỗi khi phục vụ ảnh {filename}: {str(e)}")
        return jsonify({"error": f"Lỗi khi tải ảnh: {str(e)}"}), 500

@app.teardown_appcontext
def cleanup(exception=None):
    global cap
    if cap is not None:
        logging.info("Giải phóng camera")
        cap.release()
        cap = None

if __name__ == '__main__':
    print("Đang khởi động ứng dụng...")
    logging.info("Khởi động server Flask...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)

    except Exception as e:
        logging.error(f"Lỗi khi chạy server: {e}")
        print(f"Lỗi khi chạy server: {str(e)}")
