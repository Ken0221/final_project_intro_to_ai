import cv2
from ultralytics import YOLO
import argparse
from player_reid import PlayerReID
import time

def main():
    # Camera Presets
    # Camera Presets
    CAMERA_PRESETS = {
        'default': 0,
        'pc': 0,
        'mac': 1,
        'window': 2,
        'ivcam': 2
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bowling Scoring System')
    parser.add_argument('--source', type=str, default='default', 
                        help=f'Camera source index (0, 1...) or preset name ({", ".join(CAMERA_PRESETS.keys())}) or video file path')
    args = parser.parse_args()

    # Determine source
    source = args.source
    
    # Check if source is a preset
    if source in CAMERA_PRESETS:
        source = CAMERA_PRESETS[source]
    elif source.isdigit():
        source = int(source)

    # Initialize Player Re-ID
    reid = PlayerReID()
    
    # Load the YOLOv8 model
    # 'yolov8n.pt' is the nano version, faster but less accurate. 
    # It will be downloaded automatically on first run.
    model = YOLO('yolov8n.pt')

    # Open the webcam
    print(f"Opening camera source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # COCO class ID for 'bottle' is 39
    BOTTLE_CLASS_ID = 39
    
    # Aspect ratio threshold for standing bottles (height / width)
    # Adjust this value based on your specific bottles and camera angle
    STANDING_THRESHOLD = 1.2

    print("Starting Bowling Scoring System...")
    print("Keys:")
    print("  'r': Register new player")
    print("  'q': Quit")

    # Warm-up: sometimes the first few frames are empty or the camera is initializing
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            break
        cv2.waitKey(100)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Retrying...")
            # Try to reconnect or just skip this iteration
            continue

        # Run YOLOv8 inference on the frame
        results = model(frame, verbose=False)

        # ---------------------------
        # Player Identification Logic
        # ---------------------------
        # Detect persons (class 0 in COCO)
        current_player = "Unknown"
        player_score = 0.0
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0: # Person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Crop person
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size > 0:
                        name, score = reid.identify_player(person_img)
                        
                        display_name = name if name else "Unknown"
                        if name:
                            current_player = name
                            player_score = score

                        # Draw name and box (Always draw)
                        cv2.putText(frame, f"Player: {display_name} ({score:.2f})", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                            
                        # Debug: Show what the Re-ID model sees
                        cv2.imshow('Debug: Player Crop', person_img)

        # Display Current Player at top
        cv2.putText(frame, f"Current Player: {current_player}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        standing_count = 0
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if the detected object is a bottle
                cls_id = int(box.cls[0])
                if cls_id == BOTTLE_CLASS_ID:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate width and height
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Calculate aspect ratio
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Determine if standing
                    is_standing = aspect_ratio > STANDING_THRESHOLD
                    
                    if is_standing:
                        standing_count += 1
                        color = (0, 255, 0) # Green for standing
                        label = f"Standing ({aspect_ratio:.2f})"
                    else:
                        color = (0, 0, 255) # Red for fallen
                        label = f"Fallen ({aspect_ratio:.2f})"

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Put label
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the total count
        cv2.putText(frame, f"Standing Bottles: {standing_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show the frame
        cv2.imshow('Bowling Scoring System', frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Break the loop if 'q' is pressed
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Registration Mode
            print("Enter player name in terminal:")
            # Temporarily close window or just wait?
            # cv2.waitKey(1) might not be enough to update UI if we block on input
            # Let's just block.
            player_name = input("Player Name: ")
            if player_name:
                print(f"Please stand in front of the camera for {player_name}...")
                time.sleep(2) # Give time to pose
                
                # Capture a fresh frame
                ret, reg_frame = cap.read()
                if ret:
                    # Detect person in this frame to crop
                    reg_results = model(reg_frame, verbose=False)
                    found_person = False
                    for result in reg_results:
                        for box in result.boxes:
                            if int(box.cls[0]) == 0: # Person
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                person_crop = reg_frame[y1:y2, x1:x2]
                                if person_crop.size > 0:
                                    reid.register_player(player_name, person_crop)
                                    found_person = True
                                    print("Registration successful!")
                                    break
                        if found_person: break
                    
                    if not found_person:
                        print("No person detected for registration. Try again.")
                else:
                    print("Failed to capture registration frame.")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
