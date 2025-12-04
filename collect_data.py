import cv2
import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Data Collection for Gesture Recognition')
    parser.add_argument('--source', type=str, default='0', help='Camera source index')
    args = parser.parse_args()

    # Determine source
    source = args.source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera {source}")
        return

    # Create directories if they don't exist
    base_dir = "datasets/raw"
    for i in range(1, 6):
        os.makedirs(os.path.join(base_dir, str(i)), exist_ok=True)

    current_class = 1
    count = 0
    
    # Get initial count for current class
    count = len(os.listdir(os.path.join(base_dir, str(current_class))))

    print("Starting Data Collection...")
    print("Keys:")
    print("  1-5: Switch Class")
    print("  SPACE: Save Image")
    print("  q: Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display info
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Class: {current_class}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Count: {count}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to save", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Data Collection', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            current_class = int(chr(key))
            # Update count for new class
            count = len(os.listdir(os.path.join(base_dir, str(current_class))))
            print(f"Switched to Class {current_class}")
        elif key == ord(' '):
            # Save image
            timestamp = int(time.time() * 1000)
            filename = f"{base_dir}/{current_class}/{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            count += 1
            print(f"Saved {filename}")
            # Flash effect
            cv2.imshow('Data Collection', 255 - display_frame)
            cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
