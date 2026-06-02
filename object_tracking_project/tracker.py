import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("Local file codec fallback, opening online...")
    video_path = "https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/_static/videos/vtest.avi"
    cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Still blocked. Defaulting safely to Webcam.")
    cap = cv2.VideoCapture(0)
print("Tracking initialized successfully! Press 'q' to quit.")
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    results = model.track(frame, persist=True, device="cpu", verbose=False)
    if results[0].boxes is not None and results[0].boxes.id is not None:
        current_count = len(results[0].boxes.id)
        annotated_frame = results[0].plot()
    else:
        current_count = 0
        annotated_frame = frame
    counter_text = f"Objects in Frame: {current_count}"
    cv2.rectangle(annotated_frame, (10, 10), (280, 50), (0, 0, 0), -1)
    cv2.putText(annotated_frame, counter_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("YOLOv8 Object Tracking ^& Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
