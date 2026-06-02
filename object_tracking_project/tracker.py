import cv2
from ultralytics import YOLO

def main():
    print("🚀 Initializing Native Real-Time Webcam YOLOv8 Object Tracker...")
    
    # Load the pre-trained deep learning YOLOv8 architecture weights
    model = YOLO("yolov8n.pt")
    
    # Establish a stream pointer connection with your default system webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open or establish hardware connection with webcam.")
        return

    print("✅ Live camera stream established successfully. Press 'q' to stop tracking.")
    
    # A dictionary memory cache to store bounding box positions from the previous frame
    previous_centers = {}
    next_object_id = 1

    # Enter continuous matrix ingestion loop processing frame vectors in real-time
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run native YOLOv8 predictions (This skips the buggy 'lap' dependency entirely!)
        results = model.predict(frame, verbose=False)
        
        current_centers = {}
        current_frame_objects = 0

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            current_frame_objects = len(boxes)

            for box, cls_idx in zip(boxes, clss):
                x1, y1, x2, y2 = box
                label_name = model.names[cls_idx]
                
                # Calculate the center point coordinate of the bounding box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Dynamic tracking matching logic (tracks objects based on proximity)
                assigned_id = None
                min_distance = 50  # Pixel distance matching threshold
                
                for obj_id, p_center in list(previous_centers.items()):
                    distance = ((cx - p_center[0])**2 + (cy - p_center[1])**2)**0.5
                    if distance < min_distance:
                        assigned_id = obj_id
                        min_distance = distance
                        break
                
                if assigned_id is None:
                    assigned_id = next_object_id
                    next_object_id += 1
                
                current_centers[assigned_id] = (cx, cy)

                # Render bright green bounding rectangle around targets
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Overlay tracking ID text labels right above bounding borders
                text_overlay = f"{label_name.upper()} ID: {assigned_id}"
                cv2.putText(frame, text_overlay, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        previous_centers = current_centers

        # Overlay the high-level visual statistical status counter banner in top corner
        cv2.putText(frame, f"Active Objects in View: {current_frame_objects}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        
        cv2.putText(frame, f"Total System Tracked: {next_object_id - 1}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Push processed matrices out to the graphical interface container layout window
        cv2.imshow("Live Webcam Object Detection & ID Tracking", frame)

        # Breaks the execution loop dynamically if user focuses on window and hits 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🔒 Live camera stream terminated safely.")

if __name__ == "__main__":
    main()