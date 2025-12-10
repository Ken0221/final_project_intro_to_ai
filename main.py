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
    parser.add_argument('--backbone', type=str, default='resnet101',
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
    
    # Set camera to maximum resolution to ensure largest Field of View (FOV)
    # Setting a very high value forces OpenCV/Camera to fall back to the maximum supported resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera FOV maximized. Resolution: {int(actual_width)}x{int(actual_height)}")

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

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame. Retrying...")
            continue

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
        
        # 1. Collect all potential matches
        # candidates = [(score, person_idx, player_name), ...]
        candidates = []
        person_data = [] # Stores (box, scores_dict) for rendering later
        
        ID_THRESHOLD = 0.55

        for idx, (person_img, (x1, y1, x2, y2)) in enumerate(persons):
            # We use identify_player mainly to populate scores_dict
            # We ignore the returned 'best_match' decision because we do our own greedy assignment
            _, _, scores_dict = reid.identify_player(person_img, threshold=ID_THRESHOLD)
            person_data.append(((x1, y1, x2, y2), scores_dict))
            
            if scores_dict:
                for p_name, p_vals in scores_dict.items():
                    s = p_vals['combined']
                    if s > ID_THRESHOLD:
                        candidates.append((s, idx, p_name))
        
        # 2. Global Greedy Matching
        # Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        assigned_persons = set()
        assigned_players = set()
        final_assignments = {} # person_idx -> (name, score)

        for score, p_idx, p_name in candidates:
            # If this person is already assigned, or this player is already found
            if p_idx in assigned_persons or p_name in assigned_players:
                continue
            
            # Assign
            final_assignments[p_idx] = (p_name, score)
            assigned_persons.add(p_idx)
            assigned_players.add(p_name)

        # 3. Render Results
        for idx, (box_coords, scores_dict) in enumerate(person_data):
            x1, y1, x2, y2 = box_coords
            
            if idx in final_assignments:
                name, score = final_assignments[idx]
                display_name = name
                color = (255, 0, 255) # Magenta for identified
                
                # Update "Current Player" display if it's the highest score so far in loop?? 
                # Actually usually we just want to know who is 'active'. 
                # For now let's just pick the last one or the one with highest score overall?
                # Let's keep the logic simple: update 'current_player' if this is a valid match
                # (This might toggle if multiple players are in frame, but standard behavior)
                current_player = name
                player_score = score
            else:
                display_name = "Unknown"
                score = 0.0
                color = (0, 0, 255) # Red for unknown

            # Draw name and box
            if display_name == "Unknown":
                 cv2.putText(frame, f"Player: {display_name}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                cv2.putText(frame, f"Player: {display_name} ({score:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Debug mode: show all scores on frame
            if debug_mode and scores_dict:
                y_offset = y1 + 20
                for p_name, p_scores in scores_dict.items():
                    debug_text = f"{p_name}: {p_scores['combined']:.2f}"
                    cv2.putText(frame, debug_text, (x1, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 18

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
        
        for (x1, y1, x2, y2) in bottles:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put label
            cv2.putText(frame, "Bottle", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Future work: How to determine if a bottle is standing or fallen?

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

                        # Find the person with the largest bounding box
                        best_person_crop = None
                        max_area = 0
                        
                        for person_img, (x1, y1, x2, y2) in reg_persons:
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
                                best_person_crop = person_img
                        
                        if best_person_crop is not None:
                            sample_count = reid.register_player(
                                player_name, 
                                best_person_crop,
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
