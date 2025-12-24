"""
Tennis Ball Tracking Test
==========================
A simple test script to track a tennis ball (sports ball) using
the camera tracking system with pan/tilt servo control.

Based on main.py structure, simplified for ball tracking only.

Usage:
    python track_test.py
    python track_test.py --source 0
    python track_test.py --source ivcam --com COM4

Author: AI Intro Final Project
Date: 2024-12
"""

import cv2
import argparse
import time
import torch
from arduino_control import CameraTracker
from yolo_detection import YOLODetector



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
    parser = argparse.ArgumentParser(description='Tennis Ball Tracking Test')
    parser.add_argument('--source', type=str, default='default',
                        help=f'Camera source: index or preset ({", ".join(CAMERA_PRESETS.keys())})')
    parser.add_argument('--model', type=str, default='yolo11n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                                 'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x', 'best'],
                        help='YOLO model to use')
    parser.add_argument('--com', type=str, default='COM4',
                        help='Arduino COM port')
    parser.add_argument('--baud', type=int, default=115200,
                        help='Serial baud rate')
    parser.add_argument('--target', type=str, default='tennis_ball',
                        help='Target object to track (COCO class name)')
    parser.add_argument('--show_everything', type=bool, default=False,
                        help='Show everything')

    args = parser.parse_args()

    # Determine camera source
    source = args.source
    if source in CAMERA_PRESETS:
        source = CAMERA_PRESETS[source]
    elif source.isdigit():
        source = int(source)

    # Initialize YOLO Detector (for visualization)
    # Use GPU if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading YOLO model: {args.model}.pt")
    yolo = YOLODetector(model_path=f"{args.model}.pt")

    # Open the webcam
    print(f"Opening camera source: {source}")
    cap = cv2.VideoCapture(source)
    
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000000)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Initialize Camera Tracker
    tracker = CameraTracker(
        com_port=args.com,
        baud_rate=args.baud,
        frame_width=actual_width,
        frame_height=actual_height
    )
    
    # Share the same YOLO detector to avoid loading twice
    tracker._detector = yolo

    # Update tracker frame size
    tracker.update_frame_size(actual_width, actual_height)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n" + "=" * 50)
    print(f"Tennis Ball Tracking Test")
    print("=" * 50)
    print(f"\nTarget: {args.target}")
    print(f"Arduino: {args.com} @ {args.baud} baud")
    print("\nKeys:")
    print("  'c': Connect to Arduino")
    print("  'd': Disconnect from Arduino")
    print("  'q': Quit")
    print("=" * 50 + "\n")

    # Warm-up camera
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            break
        cv2.waitKey(100)

    prev_time = 0
    target_label = args.target
    show_everything = args.show_everything

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Retrying...")
                continue

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Track the target object
            result = tracker.camera_track(frame, target_label)

            # Show all detections if enabled
            if show_everything:
                results = yolo.detect(frame)
                attrs = yolo.get_everything_attr(results)
                for cls, box_coords, _ in attrs:
                    x1, y1, x2, y2 = map(int, box_coords)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw frame center crosshair
            cx_frame = tracker.frame_center_x
            cy_frame = tracker.frame_center_y
            cv2.drawMarker(frame, (cx_frame, cy_frame), (255, 255, 255), 
                           cv2.MARKER_CROSS, 20, 2)

            # Draw dead zone rectangle (convert ratio to pixels)
            dz_x = int(tracker.dead_zone_x * tracker.frame_center_x)
            dz_y = int(tracker.dead_zone_y * tracker.frame_center_y)
            cv2.rectangle(frame,
                          (cx_frame - dz_x, cy_frame - dz_y),
                          (cx_frame + dz_x, cy_frame + dz_y),
                          (100, 100, 100), 1)

            if result['found']:
                # Draw bounding box
                x1, y1, x2, y2 = result['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw object center
                cx, cy = result['center']
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                
                # Draw line from frame center to object
                cv2.line(frame, (cx_frame, cy_frame), (cx, cy), (255, 255, 0), 2)
                
                # Label
                cv2.putText(frame, f"{target_label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Tracking info
                error_x, error_y = result['error']
                info_lines = [
                    f"Target: {target_label} FOUND",
                    f"Center: ({cx}, {cy})",
                    f"Error: ({error_x}, {error_y})",
                    f"Pan Speed: {result['pan_speed']} (90=stop)",
                    f"Tilt Angle: {result['tilt_angle']}°"
                ]
            else:
                info_lines = [
                    f"Target: {target_label} NOT FOUND",
                    f"Pan Speed: 90 (stopped)",
                    f"Tilt Angle: {tracker.current_tilt_angle}°"
                ]

            # Display info
            y_offset = 30
            cv2.putText(frame, f"FPS: {int(fps)}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
            
            for line in info_lines:
                color = (0, 255, 0) if result['found'] else (0, 0, 255)
                cv2.putText(frame, line, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25

            # Connection status
            status = "CONNECTED" if tracker.connected else "DISCONNECTED"
            status_color = (0, 255, 0) if tracker.connected else (0, 0, 255)
            cv2.putText(frame, f"Arduino: {status}", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Show frame
            cv2.imshow('Tennis Ball Tracking', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Connecting to Arduino...")
                tracker.connect()
            elif key == ord('d'):
                print("Disconnecting from Arduino...")
                tracker.disconnect()

    except KeyboardInterrupt:
        print("\nCtrl+C detected! Stopping motors...")
    
    finally:
        # Cleanup - always runs whether normal exit, 'q' press, or Ctrl+C
        tracker.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print("Tracking test ended.")


if __name__ == "__main__":
    main()
