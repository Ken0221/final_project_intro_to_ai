"""
Bowling Scorer Web Application
==============================
Flask + SocketIO server for real-time bowling game with camera tracking.

The tracking control loop runs directly in Python (no web latency affecting
servo control). The web interface only displays processed frames and handles
game operations.

Author: AI Intro Final Project
Date: 2024-12
"""

import cv2
import base64
import argparse
import time
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from game_manager import GameManager, GameState
from yolo_detection import YOLODetector
from player_reid import PlayerReID
from arduino_control import CameraTracker

# ==========================================
# Flask App Setup
# ==========================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bowling-scorer-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ==========================================
# Global State
# ==========================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.game_manager = GameManager()
        self.yolo: YOLODetector = None
        self.reid: PlayerReID = None
        self.tracker: CameraTracker = None
        self.cap: cv2.VideoCapture = None
        
        self.camera_source = 0
        self.com_port = "/dev/cu.usbmodem1201"

        # hyperparameters
        self.conf = 0.1
        self.iou = 0.7
        
        # Model settings
        self.yolo_model = "yolo11s"
        self.reid_backbone = "resnet50"
        
        self.running = False
        self.video_thread = None
        
        # Current frame data
        self.current_frame = None
        self.current_player_crop = None
        self.standing_bottles = 10
        
        # Tracking state
        self.ball_visible_count = 0
        self.ball_lost_count = 0
        self.bottles_visible_count = 0
        self.counting_start_time = None
        
        # Constants for state transitions
        self.BALL_DETECT_THRESHOLD = 3  # Frames to confirm ball detected
        self.BALL_LOST_THRESHOLD = 10   # Frames to confirm ball lost
        self.BOTTLE_DETECT_THRESHOLD = 5  # Frames to confirm bottles visible

app_state = AppState()

# ==========================================
# Routes
# ==========================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

# ==========================================
# SocketIO Events
# ==========================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('game_state', app_state.game_manager.get_game_state_dict())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('set_settings')
def handle_settings(data):
    """Update game settings"""
    app_state.com_port = data.get('com_port', '/dev/cu.usbmodem1201')
    app_state.camera_source = data.get('camera_source', '0')
    num_frames = data.get('num_frames', 10)
    num_bottles = data.get('num_bottles', 10)
    
    # Model settings
    app_state.yolo_model = data.get('yolo_model', 'yolo11n')
    app_state.reid_backbone = data.get('reid_backbone', 'resnet50')
    
    # Convert camera source
    if app_state.camera_source.isdigit():
        app_state.camera_source = int(app_state.camera_source)
    
    app_state.game_manager.set_settings(app_state.com_port, num_frames, num_bottles)
    app_state.game_manager.state = GameState.REGISTRATION
    
    # Initialize components with new settings
    initialize_components(force_reload=True)
    
    # Start video processing thread
    start_video_thread()
    
    emit('game_state', app_state.game_manager.get_game_state_dict())
    print(f"Settings: COM={app_state.com_port}, Bottles={num_bottles}, YOLO={app_state.yolo_model}")

@socketio.on('register_player')
def handle_register_player(data):
    """Register a player"""
    player_num = data.get('player_num')
    name = data.get('name', '').strip()
    
    if not name:
        emit('registration_result', {'success': False, 'message': 'Name is required'})
        return
    
    # Capture current frame for registration
    if app_state.current_frame is not None:
        frame = app_state.current_frame.copy()
        
        # Detect persons
        results = app_state.yolo.detect(frame, conf=app_state.conf, iou=app_state.iou)
        persons = app_state.yolo.get_person_crops(frame, results, conf=app_state.conf)
        
        if persons:
            # Get largest person crop
            best_crop = None
            best_area = 0
            for crop, (x1, y1, x2, y2) in persons:
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_crop = crop
            
            if best_crop is not None:
                # Register with ReID
                app_state.reid.register_player_from_samples(name, [best_crop])
                
                # Register in game manager
                if app_state.game_manager.register_player(name):
                    emit('registration_result', {
                        'success': True,
                        'player_num': player_num,
                        'name': name
                    })
                    emit('game_state', app_state.game_manager.get_game_state_dict())
                    return
        
        emit('registration_result', {
            'success': False,
            'message': 'No person detected. Please stand in front of the camera.'
        })
    else:
        emit('registration_result', {
            'success': False,
            'message': 'Camera not ready'
        })

@socketio.on('start_game')
def handle_start_game():
    """Start the game"""
    if app_state.game_manager.start_game():
        # Connect to Arduino
        if app_state.tracker:
            app_state.tracker.connect()
            emit('arduino_status', {'connected': app_state.tracker.connected})
        
        emit('game_state', app_state.game_manager.get_game_state_dict())
        print("Game started!")
    else:
        emit('error', {'message': 'Cannot start game. Need 2 players.'})

@socketio.on('correct_score')
def handle_correct_score(data):
    """Correct the last recorded score"""
    pins_knocked = data.get('pins_knocked', 0)
    result = app_state.game_manager.correct_last_throw(pins_knocked)
    
    if result:
        emit('game_state', app_state.game_manager.get_game_state_dict())
        emit('score_update', result)
    else:
        emit('error', {'message': 'Cannot correct score'})

@socketio.on('reset_game')
def handle_reset_game():
    """Reset the game for a new round"""
    app_state.game_manager.reset_game()
    app_state.standing_bottles = 10
    app_state.ball_visible_count = 0
    app_state.ball_lost_count = 0
    app_state.bottles_visible_count = 0
    
    emit('game_state', app_state.game_manager.get_game_state_dict())

@socketio.on('connect_arduino')
def handle_connect_arduino():
    """Manually connect to Arduino"""
    if app_state.tracker:
        app_state.tracker.connect()
        emit('arduino_status', {'connected': app_state.tracker.connected})

@socketio.on('disconnect_arduino')
def handle_disconnect_arduino():
    """Manually disconnect from Arduino"""
    if app_state.tracker:
        app_state.tracker.disconnect()
        emit('arduino_status', {'connected': False})

# ==========================================
# Video Processing
# ==========================================

def initialize_components(force_reload=False):
    """Initialize YOLO, ReID, Tracker, and Camera"""
    print("Initializing components...")
    
    # Initialize YOLO (reload if model changed)
    if app_state.yolo is None or force_reload:
        print(f"Loading YOLO model: {app_state.yolo_model}.pt")
        app_state.yolo = YOLODetector(model_path=f"{app_state.yolo_model}.pt")
    
    # Initialize ReID (reload if backbone changed)
    if app_state.reid is None or force_reload:
        print(f"Loading ReID model with backbone: {app_state.reid_backbone}")
        app_state.reid = PlayerReID(backbone=app_state.reid_backbone)
    
    # Initialize Camera
    if app_state.cap is None or not app_state.cap.isOpened():
        print(f"Opening camera: {app_state.camera_source}")
        app_state.cap = cv2.VideoCapture(app_state.camera_source)
        app_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000000)
        app_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000000)
        
        actual_width = int(app_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(app_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Initialize Tracker (or update detector reference)
    actual_width = int(app_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(app_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if app_state.tracker is None:
        app_state.tracker = CameraTracker(
            com_port=app_state.com_port,
            baud_rate=115200,
            frame_width=actual_width,
            frame_height=actual_height
        )
    
    # Share YOLO detector (update reference if reloaded)
    app_state.tracker._detector = app_state.yolo
    app_state.tracker.update_frame_size(actual_width, actual_height)
    
    print("Components initialized!")

def start_video_thread():
    """Start the video processing thread"""
    if app_state.video_thread is None or not app_state.video_thread.is_alive():
        app_state.running = True
        app_state.video_thread = threading.Thread(target=video_processing_loop, daemon=True)
        app_state.video_thread.start()
        print("Video processing thread started")

def video_processing_loop():
    """Main video processing loop - runs tracking at full speed"""
    print("Video processing loop started")
    
    frame_count = 0
    last_emit_time = 0
    emit_interval = 0.033  # ~30 FPS for web display
    
    while app_state.running:
        if app_state.cap is None or not app_state.cap.isOpened():
            time.sleep(0.1)
            continue
        
        ret, frame = app_state.cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        app_state.current_frame = frame.copy()
        
        # Process based on game state
        game_state = app_state.game_manager.state
        annotated_frame = frame.copy()
        
        if game_state == GameState.REGISTRATION:
            # During registration, just show detections
            annotated_frame = process_registration_frame(frame)
        
        elif game_state in (GameState.WAITING_FOR_THROW, GameState.TRACKING_PLAYER,
                           GameState.TRACKING_BALL, GameState.TRACKING_BOTTLE,
                           GameState.COUNTING):
            # During game, run tracking and game logic
            annotated_frame = process_game_frame(frame)
        
        # Emit frame to web clients (rate limited)
        current_time = time.time()
        if current_time - last_emit_time >= emit_interval:
            emit_frame(annotated_frame)
            last_emit_time = current_time

        # debug state
        # print(app_state.game_manager.get_game_state_dict().get("state"))
    
    print("Video processing loop ended")

def process_registration_frame(frame):
    """Process frame during registration phase"""
    results = app_state.yolo.detect(frame, conf=app_state.conf, iou=app_state.iou)
    persons = app_state.yolo.get_person_crops(frame, results, conf=app_state.conf)
    
    # Draw person detections
    for crop, (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person Detected", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def process_game_frame(frame):
    """Process frame during game - handle tracking and state transitions"""
    gm = app_state.game_manager
    
    # Run YOLO detection
    results = app_state.yolo.detect(frame, conf=app_state.conf, iou=app_state.iou)
    
    # Get current player info
    current_player = gm.get_current_player()
    if not current_player:
        return frame
    
    # Process based on current tracking state
    if gm.state == GameState.WAITING_FOR_THROW:
        # Waiting for throw - start tracking player
        gm.start_throw()
        socketio.emit('game_state', gm.get_game_state_dict())
        socketio.emit('tracking_status', {'target': 'person', 'found': False})
    
    if gm.state == GameState.TRACKING_PLAYER:
        frame = track_player(frame, results, current_player.name)
    
    elif gm.state == GameState.TRACKING_BALL:
        frame = track_ball(frame, results)
    
    elif gm.state == GameState.TRACKING_BOTTLE:
        frame = track_bottles(frame, results)
    
    elif gm.state == GameState.COUNTING:
        frame = count_bottles(frame, results)
    
    return frame

def track_player(frame, results, player_name):
    """Track the current player"""
    gm = app_state.game_manager
    
    # Get person crops and identify player
    persons = app_state.yolo.get_person_crops(frame, results, conf=app_state.conf)
    
    target_box = None
    for crop, (x1, y1, x2, y2) in persons:
        identified, score, _ = app_state.reid.identify_player(crop, threshold=0.5)
        
        # if identified == player_name:
        if True:
            target_box = (x1, y1, x2, y2)
            app_state.current_player_crop = crop
            
            # Draw player box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, f"{player_name} ({score:.2f})", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            break
    
    # Track player with gimbal
    if target_box and app_state.tracker and app_state.tracker.connected:
        cx = (target_box[0] + target_box[2]) // 2
        cy = (target_box[1] + target_box[3]) // 2
        
        # Calculate error and send to tracker
        error_x = (app_state.tracker.frame_center_x - cx) / app_state.tracker.frame_center_x
        error_y = (app_state.tracker.frame_center_y - cy) / app_state.tracker.frame_center_y
        # print("Error X:", error_x)
        # print("Error Y:", error_y)
        pan_speed = app_state.tracker._calculate_pan_speed(error_x, is_tennis=False)
        new_tilt_angle = app_state.tracker._calculate_tilt_angle(error_y)
        # print("Pan Speed:", pan_speed)
        # print("New Tilt Angle:", new_tilt_angle)
        # Update tracker state (critical for incremental tilt adjustment!)
        app_state.tracker.current_tilt_angle = new_tilt_angle
        
        # print(f"Pan Speed: {pan_speed}, Tilt Angle: {tilt_angle}")
        app_state.tracker._send_command(pan_speed, new_tilt_angle)
    elif app_state.tracker and app_state.tracker.connected:
        # Target not found - stop pan motor, maintain tilt
        app_state.tracker._send_command(90, app_state.tracker.current_tilt_angle)
    
    # Emit player crop
    if app_state.current_player_crop is not None:
        emit_player_crop(app_state.current_player_crop, player_name)
    
    # Check for ball - transition to TRACKING_BALL
    all_objects = app_state.yolo.get_everything_attr(results, conf=app_state.conf)
    ball_found = any(cls.lower() == 'tennisball' for cls, _, _ in all_objects)
    
    if ball_found:
        app_state.ball_visible_count += 1
        if app_state.ball_visible_count >= app_state.BALL_DETECT_THRESHOLD:
            gm.on_ball_detected()
            app_state.ball_visible_count = 0
            socketio.emit('tracking_status', {'target': 'tennisball', 'found': True})
            socketio.emit('game_state', gm.get_game_state_dict())
    else:
        app_state.ball_visible_count = 0
    
    return frame

def track_ball(frame, results):
    """Track the ball"""
    gm = app_state.game_manager
    
    # Track ball with gimbal
    if app_state.tracker:
        result = app_state.tracker.camera_track(frame, "tennisball", conf=app_state.conf)
        
        if result['found']:
            app_state.ball_lost_count = 0
            x1, y1, x2, y2 = result['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, "Ball", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            app_state.ball_lost_count += 1
    
    # Check for bottles - transition to TRACKING_BOTTLE
    bottles = get_standing_bottles(results)
    
    if len(bottles) > 0:
        app_state.bottles_visible_count += 1
    else:
        app_state.bottles_visible_count = 0
    
    # Transition conditions:
    # 1. Ball lost AND bottles visible
    # 2. Ball lost for too long
    if app_state.ball_lost_count >= app_state.BALL_LOST_THRESHOLD:
        if app_state.bottles_visible_count >= app_state.BOTTLE_DETECT_THRESHOLD:
            gm.on_bottles_visible()
            app_state.counting_start_time = time.time()
            socketio.emit('tracking_status', {'target': 'plastic-bottle', 'found': True})
            socketio.emit('game_state', gm.get_game_state_dict())
    
    return frame

def track_bottles(frame, results):
    """Track bottles and start counting"""
    gm = app_state.game_manager
    
    # Track bottles with gimbal
    if app_state.tracker:
        app_state.tracker.camera_track(frame, "plastic-bottle", conf=app_state.conf)
    
    # # Draw bottles
    # bottles = get_standing_bottles(results)
    # for (x1, y1, x2, y2) in bottles:
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, "Standing", (x1, y1 - 10),
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Check countdown
    if app_state.counting_start_time:
        elapsed = time.time() - app_state.counting_start_time
        remaining = max(0, 3 - int(elapsed))
        
        socketio.emit('countdown', {'seconds': remaining})
        
        if elapsed >= 3.0:
            gm.state = GameState.COUNTING
            socketio.emit('countdown', {'seconds': 0})
    
    return frame

def count_bottles(frame, results):
    """Count standing bottles and record throw"""
    gm = app_state.game_manager
    
    bottles = get_standing_bottles(results)
    standing_count = len(bottles)
    
    # Draw bottles
    for (x1, y1, x2, y2) in bottles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display count
    cv2.putText(frame, f"Standing: {standing_count}", (50, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    # Record the throw
    result = gm.record_throw(standing_count)
    
    # Emit updates
    socketio.emit('score_update', result)
    socketio.emit('game_state', gm.get_game_state_dict())
    
    # Reset tracking state
    app_state.ball_visible_count = 0
    app_state.ball_lost_count = 0
    app_state.bottles_visible_count = 0
    app_state.counting_start_time = None
    
    # Check game over
    if result.get('game_over'):
        socketio.emit('game_over', gm.get_game_state_dict())
    
    return frame

def get_standing_bottles(results):
    """Get standing bottles from YOLO results"""
    bottles = app_state.yolo.get_bottles(results)
    standing = []
    
    STANDING_THRESHOLD = 1.2  # height/width ratio
    
    for (x1, y1, x2, y2) in bottles:
        width = x2 - x1
        height = y2 - y1
        if width > 0:
            ratio = height / width
            if ratio > STANDING_THRESHOLD:
                standing.append((x1, y1, x2, y2))
    
    return standing

def emit_frame(frame):
    """Encode and emit frame to clients"""
    try:
        # Resize for web display
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        socketio.emit('video_frame', {'frame': frame_base64})
    except Exception as e:
        print(f"Error emitting frame: {e}")

def emit_player_crop(crop, player_name):
    """Encode and emit player crop to clients"""
    try:
        # Resize crop
        crop_resized = cv2.resize(crop, (150, 200))
        
        _, buffer = cv2.imencode('.jpg', crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
        crop_base64 = base64.b64encode(buffer).decode('utf-8')
        
        socketio.emit('player_crop', {'crop': crop_base64, 'player_name': player_name})
    except Exception as e:
        print(f"Error emitting player crop: {e}")

# ==========================================
# Main Entry Point
# ==========================================

def cleanup():
    """Clean up resources"""
    print("Cleaning up...")
    app_state.running = False
    
    if app_state.tracker:
        app_state.tracker.disconnect()
    
    if app_state.cap:
        app_state.cap.release()
    
    print("Cleanup complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bowling Scorer Web App')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    args = parser.parse_args()
    
    try:
        print("=" * 50)
        print("🎳 Bowling Scorer Web Application")
        print("=" * 50)
        print(f"Starting server on http://{args.host}:{args.port}")
        print("Open a browser to access the application")
        print("=" * 50)
        
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()
