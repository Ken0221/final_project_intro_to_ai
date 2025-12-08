import cv2
import argparse
from player_reid import PlayerReID
from yolo_detection import YOLODetector
import time

def main():
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
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                                 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x'],
                        help='YOLO model to use')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'convnext_small', 'mobilenetv2'],
                        help='Backbone model for player re-identification')
    parser.add_argument('--debug_mode', type=bool, default=False,
                        help='Debug mode')
    parser.add_argument('--show_everything', type=bool, default=False,
                        help='Show everything')
    args = parser.parse_args()

    # State variables
    debug_mode = args.debug_mode
    show_everything = args.show_everything

    # Determine source
    source = args.source
    
    # Check if source is a preset
    if source in CAMERA_PRESETS:
        source = CAMERA_PRESETS[source]
    elif source.isdigit():
        source = int(source)

    # Initialize Player Re-ID with selected backbone
    reid = PlayerReID(backbone=args.backbone)
    
    # Initialize YOLO Detector
    yolo = YOLODetector(model_path=f"{args.model}.pt")

    # Open the webcam
    print(f"Opening camera source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Aspect ratio threshold for standing bottles (height / width)
    # Adjust this value based on your specific bottles and camera angle
    STANDING_THRESHOLD = 1.2

    print("\n" + "="*50)
    print("Starting Bowling Scoring System...")
    print("="*50)
    print("\nKeys:")
    print("  'r': Register new player (can register same player multiple times)")
    print("  'd': Toggle Debug mode (show all similarity scores)")
    print("  'l': List all registered players")
    print("  'c': Delete (Clear) a player")
    print("  'q': Quit")
    print("="*50 + "\n")

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
        results = yolo.detect(frame)

        if show_everything:
            attrs = yolo.get_everything_attr(results)
            for cls, box_coords in attrs:
                x1, y1, x2, y2 = map(int, box_coords)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # ---------------------------
        # Player Identification Logic
        # ---------------------------
        current_player = "Unknown"
        player_score = 0.0
        
        # Get persons using helper
        persons = yolo.get_person_crops(frame, results)
        
        for person_img, (x1, y1, x2, y2) in persons:
            name, score, scores_dict = reid.identify_player(person_img)
            
            display_name = name if name else "Unknown"
            if name:
                current_player = name
                player_score = score

            # Draw name and box (Always draw)
            cv2.putText(frame, f"Player: {display_name} ({score:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Debug mode: show all scores on frame
            if debug_mode and scores_dict:
                y_offset = y1 + 20
                for p_name, p_scores in scores_dict.items():
                    debug_text = f"{p_name}: {p_scores['combined']:.2f} (max:{p_scores['max']:.2f}, mean:{p_scores['mean']:.2f})"
                    cv2.putText(frame, debug_text, (x1, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 18
                
            # Debug: Show what the Re-ID model sees
            # cv2.imshow('Debug: Player Crop', person_img)

        # Display Current Player at top
        cv2.putText(frame, f"Current Player: {current_player}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Show debug mode indicator
        if debug_mode:
            cv2.putText(frame, "[DEBUG MODE]", (20, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        standing_count = 0
        
        # ---------------------------
        # Bottle Detection Logic
        # ---------------------------
        bottles = yolo.get_bottles(results)
        
        for box in bottles:
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
            print("\n" + "="*40)
            print("PLAYER REGISTRATION")
            print("="*40)
            print("Enter player name in terminal:")
            print("(Enter existing name to add more samples)")
            player_name = input("Player Name: ").strip()
            if player_name:
                print(f"\nPlease stand in front of the camera for {player_name}...")
                print("Capturing multiple frames for better accuracy...")
                time.sleep(2)  # Give time to pose
                
                # Capture a fresh frame
                ret, reg_frame = cap.read()
                if ret:
                    # Detect person in this frame to crop
                    reg_results = yolo.detect(reg_frame)
                    reg_persons = yolo.get_person_crops(reg_frame, reg_results)
                    
                    if reg_persons:
                        # Use the first person found
                        person_crop, _ = reg_persons[0]
                        
                        sample_count = reid.register_player(
                            player_name, 
                            person_crop, 
                            capture_multiple=True,
                            cap=cap,
                            model=yolo.model 
                        )
                        print(f"\nRegistration successful! Added {sample_count} samples.")
                        print("="*40 + "\n")
                    else:
                        print("No person detected for registration. Try again.")
                else:
                    print("Failed to capture registration frame.")
        
        elif key == ord('d'):
            # Toggle debug mode
            debug_mode = not debug_mode
            reid.set_debug_mode(debug_mode)
        
        elif key == ord('l'):
            # List all registered players
            reid.list_players()
        
        elif key == ord('c'):
            # Delete a player
            print("\n" + "="*40)
            print("DELETE PLAYER")
            print("="*40)
            reid.list_players()
            player_name = input("Enter player name to delete (or blank to cancel): ").strip()
            if player_name:
                confirm = input(f"Are you sure you want to delete '{player_name}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    reid.delete_player(player_name)
            print("="*40 + "\n")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
