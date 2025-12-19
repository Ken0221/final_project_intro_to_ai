"""
Arduino Control Module for Object Tracking
============================================
This module provides a camera tracking system that uses YOLO object detection
to track specified objects and control pan/tilt servos via Arduino.

Hardware Specifications:
- Pan Servo (Pin D9): MG996R 360° Continuous Rotation
  - Logic: 90 = Stop, <90 = Spin one way, >90 = Spin other way
- Tilt Servo (Pin D10): MG996R 180° Standard
  - Logic: Absolute angle control (0-180°)

Author: AI Intro Final Project
Date: 2024-12
"""

import serial
import time
from yolo_detection import YOLODetector


class CameraTracker:
    """
    A class to manage camera tracking with pan/tilt servo control.
    
    Maintains serial connection state and current tilt angle for smooth
    incremental movements.
    """
    
    def __init__(self, com_port='COM4', baud_rate=115200, frame_width=640, frame_height=480):
        """
        Initialize the camera tracker.
        
        Args:
            com_port (str): COM port for Arduino connection
            baud_rate (int): Serial baud rate (must match Arduino)
            frame_width (int): Width of the video frame in pixels
            frame_height (int): Height of the video frame in pixels
        """
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Calculate frame center
        self.frame_center_x = frame_width // 2
        self.frame_center_y = frame_height // 2
        
        # State management for tilt servo (absolute position)
        self.current_tilt_angle = 90  # Start at center position (0-180)
        
        # Dead zone settings (as proportion of frame size) - motor stops when within this range
        self.dead_zone_ratio_x = 0.05  # 5% of frame width
        self.dead_zone_ratio_y = 0.05  # 5% of frame height
        self.dead_zone_x = int(frame_width * self.dead_zone_ratio_x)
        self.dead_zone_y = int(frame_height * self.dead_zone_ratio_y)
        
        # Speed/angle adjustment settings
        self.pan_speed_factor = 0.2   # How much error affects pan speed
        self.tilt_step = 1            # Degrees to move per frame for tilt
        self.max_pan_deviation = 80   # Max deviation from 90 (stop) for pan speed
        
        # Tilt limits
        self.tilt_min = 30   # Minimum tilt angle (looking up)
        self.tilt_max = 150  # Maximum tilt angle (looking down)
        
        # Serial connection
        self.arduino = None
        self.connected = False
        
        # Rate limiting - don't send commands too fast
        self.last_command_time = 0
        self.min_command_interval = 0.05  # 50ms between commands (20 Hz max)
        
        # Last sent values - avoid sending same command repeatedly
        self.last_pan_speed = 90
        self.last_tilt_angle = 90
        
        # YOLO Detector - Use singleton pattern to avoid reloading
        self._detector = None
    
    @property
    def detector(self):
        """Lazy-load YOLO detector."""
        if self._detector is None:
            self._detector = YOLODetector()
        return self._detector
    
    def connect(self):
        """
        Establish connection to Arduino.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to Arduino on {self.com_port}...")
            self.arduino = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            
            # Wait for Arduino to reset after connection
            time.sleep(2)
            
            # Clear input buffer
            self.arduino.reset_input_buffer()
            
            self.connected = True
            print(f"Connected successfully! Baud rate: {self.baud_rate}")
            
            # Read Arduino ready message
            time.sleep(0.1)
            while self.arduino.in_waiting:
                response = self.arduino.readline().decode().strip()
                if response:
                    print(f"Arduino: {response}")
            
            # Send initial position (stop pan, center tilt)
            self._send_command(90, self.current_tilt_angle)
            
            return True
            
        except serial.SerialException as e:
            print(f"Failed to connect to Arduino: {e}")
            print("Tips:")
            print("  1. Check USB connection")
            print("  2. Verify COM port (Device Manager)")
            print("  3. Close Arduino IDE Serial Monitor")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close Arduino connection."""
        if self.arduino and self.arduino.is_open:
            # Stop pan motor and center tilt before disconnecting
            self._send_command(90, 90)
            time.sleep(0.1)
            self.arduino.close()
            self.connected = False
            print("Arduino disconnected.")
    
    def _send_command(self, pan_speed, tilt_angle, force=False):
        """
        Send pan and tilt values to Arduino with rate limiting.
        
        Args:
            pan_speed (int): Speed value for pan servo (0-180, 90=stop)
            tilt_angle (int): Position angle for tilt servo (0-180)
            force (bool): Force send even if rate limited
        
        Returns:
            bool: True if command sent successfully
        """
        if not self.connected or not self.arduino:
            return False
        
        # Constrain values
        pan_speed = max(0, min(180, int(pan_speed)))
        tilt_angle = max(0, min(180, int(tilt_angle)))
        
        # Check rate limiting (unless forced)
        current_time = time.time()
        if not force:
            if current_time - self.last_command_time < self.min_command_interval:
                return True  # Skip this command, too soon
            
            # Skip if same as last command (no change needed)
            if pan_speed == self.last_pan_speed and tilt_angle == self.last_tilt_angle:
                return True
        
        try:
            # Check if port is still open
            if not self.arduino.is_open:
                self.connected = False
                return False
            
            # Clear output buffer to prevent buildup
            self.arduino.reset_output_buffer()
            
            # Format: <PanSpeed,TiltAngle>\n
            command = f"<{pan_speed},{tilt_angle}>\n"
            self.arduino.write(command.encode('utf-8'))
            self.arduino.flush()  # Ensure data is sent
            
            # Update last command state
            self.last_command_time = current_time
            self.last_pan_speed = pan_speed
            self.last_tilt_angle = tilt_angle
            
            # Clear any pending input to prevent buffer overflow
            if self.arduino.in_waiting > 100:
                self.arduino.reset_input_buffer()
            
            return True
            
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def _calculate_pan_speed(self, error_x):
        """
        Calculate pan motor speed based on horizontal error.
        
        For continuous rotation servo:
        - 90 = Stop
        - < 90 = Rotate left (CCW)
        - > 90 = Rotate right (CW)
        
        Args:
            error_x (int): Horizontal error (frame_center_x - object_x)
                          Positive = object is to the left
                          Negative = object is to the right
        
        Returns:
            int: Pan speed value (0-180)
        """
        # Check dead zone - if centered enough, stop motor
        if abs(error_x) < self.dead_zone_x:
            return 90  # Stop
        
        # Map error to speed deviation
        # Larger error = faster rotation
        speed_deviation = error_x * self.pan_speed_factor
        
        # Constrain deviation
        speed_deviation = max(-self.max_pan_deviation, 
                             min(self.max_pan_deviation, speed_deviation))
        
        # Calculate final speed (centered at 90)
        # Positive error (object left) -> speed > 90 (rotate left to track)
        # Negative error (object right) -> speed < 90 (rotate right to track)
        pan_speed = 90 + speed_deviation
        
        return int(pan_speed)
    
    def _calculate_tilt_angle(self, error_y):
        """
        Calculate new tilt angle based on vertical error.
        
        Uses incremental updates to smoothly track objects.
        
        Args:
            error_y (int): Vertical error (frame_center_y - object_y)
                          Positive = object is above center
                          Negative = object is below center
        
        Returns:
            int: New tilt angle (constrained to tilt_min - tilt_max)
        """
        # Check dead zone - if centered enough, maintain current angle
        if abs(error_y) < self.dead_zone_y:
            return self.current_tilt_angle
        
        # Incremental adjustment
        # Positive error (object above) -> decrease angle (tilt up)
        # Negative error (object below) -> increase angle (tilt down)
        if error_y > 0:
            new_angle = self.current_tilt_angle - self.tilt_step
        else:
            new_angle = self.current_tilt_angle + self.tilt_step
        
        # Constrain to limits
        new_angle = max(self.tilt_min, min(self.tilt_max, new_angle))
        
        return int(new_angle)
    
    def camera_track(self, frame, target_label):
        """
        Main tracking function - detect object and control servos.
        
        Args:
            frame: OpenCV frame (BGR image)
            target_label (str): Target object class to track (e.g., "person", "bottle")
        
        Returns:
            dict: Tracking result with keys:
                - 'found': bool, whether target was found
                - 'center': tuple (cx, cy) if found, None otherwise
                - 'box': tuple (x1, y1, x2, y2) if found, None otherwise
                - 'pan_speed': int, current pan speed sent
                - 'tilt_angle': int, current tilt angle sent
                - 'error': tuple (error_x, error_y) if found, None otherwise
        """
        result = {
            'found': False,
            'center': None,
            'box': None,
            'pan_speed': 90,
            'tilt_angle': self.current_tilt_angle,
            'error': None
        }
        
        # Run YOLO detection
        detection_results = self.detector.detect(frame)
        
        # Get all detected objects with their attributes
        all_objects = self.detector.get_everything_attr(detection_results)
        
        # Search for target object
        target_box = None
        for class_name, box_coords in all_objects:
            if class_name.lower() == target_label.lower():
                target_box = box_coords
                break  # Use first match
        
        if target_box is None:
            # Target not found - stop pan motor, maintain tilt
            if self.connected:
                self._send_command(90, self.current_tilt_angle)
            return result
        
        # Calculate object center
        x1, y1, x2, y2 = map(int, target_box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Calculate errors (distance from frame center to object center)
        error_x = self.frame_center_x - cx  # Positive = object is left
        error_y = self.frame_center_y - cy  # Positive = object is above
        
        # Calculate motor commands
        pan_speed = self._calculate_pan_speed(error_x)
        new_tilt_angle = self._calculate_tilt_angle(error_y)
        
        # Update state
        self.current_tilt_angle = new_tilt_angle
        
        # Send command to Arduino
        if self.connected:
            self._send_command(pan_speed, new_tilt_angle)
        
        # Update result
        result['found'] = True
        result['center'] = (cx, cy)
        result['box'] = (x1, y1, x2, y2)
        result['pan_speed'] = pan_speed
        result['tilt_angle'] = new_tilt_angle
        result['error'] = (error_x, error_y)
        
        return result
    
    def update_frame_size(self, width, height):
        """Update frame dimensions (call if camera resolution changes)."""
        self.frame_width = width
        self.frame_height = height
        self.frame_center_x = width // 2
        self.frame_center_y = height // 2
        # Recalculate dead zones based on new frame size
        self.dead_zone_x = int(width * self.dead_zone_ratio_x)
        self.dead_zone_y = int(height * self.dead_zone_ratio_y)


# Global tracker instance for simple function-based usage
_global_tracker = None


def get_tracker(com_port='COM4', baud_rate=115200, frame_width=640, frame_height=480):
    """
    Get or create the global tracker instance.
    
    Args:
        com_port (str): COM port for Arduino
        baud_rate (int): Serial baud rate
        frame_width (int): Video frame width
        frame_height (int): Video frame height
    
    Returns:
        CameraTracker: The global tracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CameraTracker(com_port, baud_rate, frame_width, frame_height)
    return _global_tracker


def camera_track(frame, target_label, auto_connect=True):
    """
    Convenient function to track objects without managing tracker instance.
    
    Args:
        frame: OpenCV frame (BGR image)
        target_label (str): Target object class to track
        auto_connect (bool): Automatically connect to Arduino if not connected
    
    Returns:
        dict: Tracking result (see CameraTracker.camera_track)
    """
    tracker = get_tracker()
    
    if auto_connect and not tracker.connected:
        tracker.connect()
    
    return tracker.camera_track(frame, target_label)


# Demo/Test code
if __name__ == "__main__":
    import cv2
    
    print("=" * 50)
    print("Camera Tracking Demo")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Press 'c' to connect to Arduino")
    print("Press 'd' to disconnect from Arduino")
    print("=" * 50)
    
    # Create tracker
    tracker = CameraTracker(
        com_port='COM4',
        baud_rate=115200,
        frame_width=640,
        frame_height=480
    )
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Target to track
    target = "person"  # Change to "bottle" or other COCO classes
    
    print(f"\nTracking target: {target}")
    print("Connect to Arduino with 'c' key when ready.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track target
        result = tracker.camera_track(frame, target)
        
        # Draw tracking info on frame
        if result['found']:
            # Draw bounding box
            x1, y1, x2, y2 = result['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cx, cy = result['center']
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw frame center
            cv2.circle(frame, (tracker.frame_center_x, tracker.frame_center_y), 
                      5, (255, 0, 0), -1)
            
            # Draw line from center to object
            cv2.line(frame, (tracker.frame_center_x, tracker.frame_center_y),
                    (cx, cy), (255, 255, 0), 2)
            
            # Display info
            info_text = f"Pan: {result['pan_speed']} | Tilt: {result['tilt_angle']}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            error_text = f"Error: ({result['error'][0]}, {result['error'][1]})"
            cv2.putText(frame, error_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Searching for {target}...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Connection status
        status = "Connected" if tracker.connected else "Disconnected"
        cv2.putText(frame, f"Arduino: {status}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Camera Tracking", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            tracker.connect()
        elif key == ord('d'):
            tracker.disconnect()
    
    # Cleanup
    tracker.disconnect()
    cap.release()
    cv2.destroyAllWindows()
