from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Initialize camera
cam = cv2.VideoCapture(0)

# Default contour color (red)
contour_color = (0, 0, 255)

def motion_detection():
    global contour_color
    
    # Capture initial frames
    ret, frame = cam.read()
    ret1, frame1 = cam.read()

    while cam.isOpened():
        # Capture frames
        ret, frame = cam.read()
        ret1, frame1 = cam.read()

        # Perform motion detection and processing
        diff = cv2.absdiff(frame, frame1)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.absdiff(gray_diff, gray_frame)
        blurred_diff = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 1)
        _, thresh_inv = cv2.threshold(blurred_diff, 25, 255, cv2.THRESH_BINARY_INV)
        _, thresh = cv2.threshold(blurred_diff, 20, 255, cv2.THRESH_BINARY)
        dilated_inv = cv2.dilate(thresh_inv, None, iterations=1)
        dilated = cv2.dilate(thresh, None, iterations=1)
        abs_diff = cv2.absdiff(dilated_inv, dilated)
        contours_inv, _ = cv2.findContours(dilated_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on frames
        cv2.drawContours(frame1, contours_inv, -1, contour_color, 2)
        cv2.drawContours(frame, contours, -1, contour_color, 1)

      

        for c in contours:
            if cv2.contourArea(c) > 100000000000:
                # Very large movement detected
                pass

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2, 3)

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame1)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(motion_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_color/<int:r>/<int:g>/<int:b>')
def change_color(r, g, b):
    global contour_color
    contour_color = (r, g, b)
    return f'Contour color changed to RGB({r}, {g}, {b})'

if __name__ == '__main__':
    app.run(debug=True)
