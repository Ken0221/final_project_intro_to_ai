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
        self.com_port = "/dev/cu.usbmodem11201"
        
        # Model settings
        self.yolo_model = "yolo11s"
        self.reid_backbone = "resnet50"
        
        self.running = False
        self.video_thread = None

        # hyperparameter
        self.conf = 0.1
        self.iou = 0.55
        
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
        
        # Registration State - stores crops per player per angle
        # {player_num: {'front': crop, 'back': crop, 'left': crop, 'right': crop}}
        self.pending_registrations = {}
        
        # Debug mode - shows all detected persons with confidence scores
        self.debug_mode = False
        
        # Temporal Smoothing for bottle counting
        self.temporal_smoothing_enabled = True  # Default enabled
        self.bottle_count_history = []  # Rolling window of bottle counts
        self.TEMPORAL_SMOOTHING_WINDOW = 5  # Number of frames to consider
        
        # Strike detection when no bottles visible
        self.player_left_time = None  # When player left frame (for no-bottle strike detection)
        self.NO_BOTTLE_STRIKE_TIMEOUT = 5.0  # Seconds to wait before declaring strike (no bottles case)
        self.BOTTLE_COUNTING_TIMEOUT = 3.0  # Seconds to wait before counting bottles (bottles visible case)
        
        # Flag to ensure player is seen before starting next throw
        self.player_seen_after_count = False  # Start false, set true after count, set false when player seen

app_state = AppState()

# ==========================================
# Routes
# ==========================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/registered_players')
def get_registered_players():
    """Get list of registered players from player_db folder"""
    import os
    players = []
    db_path = 'player_db'
    if os.path.exists(db_path):
        for filename in os.listdir(db_path):
            if filename.endswith('.npy'):
                player_name = os.path.splitext(filename)[0]
                players.append(player_name)
    return jsonify({'players': sorted(players)})

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
    app_state.yolo_model = data.get('yolo_model', 'yolo11m')
    app_state.reid_backbone = data.get('reid_backbone', 'resnet50')
    
    # Convert camera source
    if app_state.camera_source.isdigit():
        app_state.camera_source = int(app_state.camera_source)
    
    app_state.game_manager.set_settings(app_state.com_port, num_frames, num_bottles)
    app_state.game_manager.state = GameState.REGISTRATION
    
    # Stop existing video thread before reinitializing
    stop_video_thread()
    
    # Clear pending registrations (starting fresh)
    app_state.pending_registrations = {}
    
    # Initialize components with new settings
    initialize_components(force_reload=True)
    
    # Start video processing thread
    start_video_thread()
    
    emit('game_state', app_state.game_manager.get_game_state_dict())
    print(f"Settings: COM={app_state.com_port}, Bottles={num_bottles}, YOLO={app_state.yolo_model}")

@socketio.on('register_capture')
def handle_register_capture(data):
    """Capture a single angle for registration"""
    player_num = data.get('player_num')  # e.g., 'player1' or 1
    angle = data.get('angle')            # 'front', 'back', 'left', 'right'
    
    if not player_num or not angle:
        return

    if app_state.current_frame is not None:
        frame = app_state.current_frame.copy()
        
        # Detect person
        results = app_state.yolo.detect(frame, conf=app_state.conf, iou=app_state.iou)
        persons = app_state.yolo.get_person_crops(frame, results)
        
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
                # Initialize list if needed
                if player_num not in app_state.pending_registrations:
                    app_state.pending_registrations[player_num] = []
                
                # Store crop (append to list)
                app_state.pending_registrations[player_num].append(best_crop)
                
                emit('capture_result', {
                    'success': True,
                    'player_num': player_num,
                    'angle': angle
                })
                return
        
        emit('capture_result', {
            'success': False,
            'player_num': player_num,
            'message': 'No person detected'
        })
    else:
        emit('capture_result', {
            'success': False,
            'player_num': player_num,
            'message': 'Camera not ready'
        })

@socketio.on('finalize_registration')
def handle_finalize_registration(data):
    """Finalize registration after all angles captured"""
    player_num = data.get('player_num')
    name = data.get('name', '').strip()
    
    if not name:
        emit('registration_result', {'success': False, 'message': 'Name is required'})
        return
        
    crops = app_state.pending_registrations.get(player_num, [])
    if not crops:
        emit('registration_result', {'success': False, 'message': 'No samples captured'})
        return

    # Register with ReID
    app_state.reid.register_player_from_samples(name, crops)
    
    # Register in game manager
    if app_state.game_manager.register_player(name):
        # Clear pending
        del app_state.pending_registrations[player_num]
        
        emit('registration_result', {
            'success': True,
            'player_num': player_num,
            'name': name
        })
        emit('game_state', app_state.game_manager.get_game_state_dict())
    else:
        emit('registration_result', {'success': False, 'message': 'Registration failed'})

@socketio.on('select_registered_player')
def handle_select_registered_player(data):
    """Select a previously registered player directly (no re-capture needed)"""
    player_num = data.get('player_num')
    name = data.get('name', '').strip()
    
    if not name:
        emit('registration_result', {'success': False, 'message': 'Player name is required'})
        return
    
    # Check if player exists in ReID database
    if app_state.reid is None:
        emit('registration_result', {'success': False, 'message': 'ReID not initialized. Please start setup first.'})
        return
    
    if name not in app_state.reid.known_players:
        emit('registration_result', {'success': False, 'message': f'Player "{name}" not found in database'})
        return
    
    # Register in game manager (player features already loaded from disk)
    if app_state.game_manager.register_player(name):
        emit('registration_result', {
            'success': True,
            'player_num': player_num,
            'name': name,
            'from_database': True
        })
        emit('game_state', app_state.game_manager.get_game_state_dict())
        print(f"Player {player_num} selected from database: {name}")
    else:
        emit('registration_result', {'success': False, 'message': 'Registration failed - player may already be registered'})

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
    app_state.pending_registrations = {}
    
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

@socketio.on('toggle_temporal_smoothing')
def handle_toggle_temporal_smoothing():
    """Toggle temporal smoothing for bottle detection"""
    app_state.temporal_smoothing_enabled = not app_state.temporal_smoothing_enabled
    status = "ON" if app_state.temporal_smoothing_enabled else "OFF"
    print(f"Temporal Smoothing: {status}")
    emit('temporal_smoothing_status', {'enabled': app_state.temporal_smoothing_enabled})

# ==========================================
# Video Processing
# ==========================================

def stop_video_thread():
    """Stop the video processing thread safely"""
    if app_state.video_thread is not None and app_state.video_thread.is_alive():
        print("Stopping video processing thread...")
        app_state.running = False
        # Wait for thread to finish (with timeout)
        app_state.video_thread.join(timeout=3.0)
        if app_state.video_thread.is_alive():
            print("Warning: Video thread did not stop in time")
        else:
            print("Video processing thread stopped")
        app_state.video_thread = None

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
    
    # Initialize Camera - release old camera first if force_reload
    if force_reload and app_state.cap is not None:
        print("Releasing old camera...")
        app_state.cap.release()
        app_state.cap = None
    
    if app_state.cap is None or not app_state.cap.isOpened():
        print(f"Opening camera: {app_state.camera_source}")
        app_state.cap = cv2.VideoCapture(app_state.camera_source)
        app_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000000)
        app_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000000)
        
        actual_width = int(app_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(app_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
    
    # Get current camera dimensions
    actual_width = int(app_state.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(app_state.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize Tracker - rebuild if force_reload or settings changed
    if force_reload and app_state.tracker is not None:
        print("Disconnecting old tracker...")
        app_state.tracker.disconnect()
        app_state.tracker = None
    
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
    persons = app_state.yolo.get_person_crops(frame, results)
    
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

def track_player(frame, results, target_player_name):
    """Track player using Global Greedy Matching"""
    gm = app_state.game_manager
    
    # Get person crops
    persons = app_state.yolo.get_person_crops(frame, results)
    
    # 1. Collect all potential matches
    candidates = [] # [(score, person_idx, player_name), ...]
    person_data = [] # Stores (box, scores_dict) for rendering
    
    ID_THRESHOLD = 0.55
    
    for idx, (person_img, (x1, y1, x2, y2)) in enumerate(persons):
        _, _, scores_dict = app_state.reid.identify_player(person_img, threshold=ID_THRESHOLD)
        person_data.append(((x1, y1, x2, y2), scores_dict))
        
        if scores_dict:
            for p_name, p_vals in scores_dict.items():
                s = p_vals['combined']
                if s > ID_THRESHOLD:
                    candidates.append((s, idx, p_name))
    
    # 2. Global Greedy Matching
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    assigned_persons = set()
    assigned_players = set()
    final_assignments = {} # person_idx -> (name, score)
    
    for score, p_idx, p_name in candidates:
        if p_idx in assigned_persons or p_name in assigned_players:
            continue
        
        final_assignments[p_idx] = (p_name, score)
        assigned_persons.add(p_idx)
        assigned_players.add(p_name)
    
    # 3. Find target player and visualize
    target_box = None
    
    for idx, (box_coords, scores_dict) in enumerate(person_data):
        x1, y1, x2, y2 = box_coords
        
        display_name = "Unknown"
        score = 0.0
        color = (0, 0, 255) # Red for unknown
        
        if idx in final_assignments:
            name, s = final_assignments[idx]
            display_name = name
            score = s
            color = (255, 0, 255) # Magenta for identified
            
            if name == target_player_name:
                target_box = (x1, y1, x2, y2)
                app_state.current_player_crop = persons[idx][0]
        
        # Draw on frame
        # Debug mode: show ALL persons with their scores
        # Normal mode: only show target player
        if app_state.debug_mode or display_name == target_player_name:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display_name} ({score:.2f})", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Debug mode: also show all player scores for this person
        if app_state.debug_mode and scores_dict:
            y_offset = y1 + 20
            for p_name, p_scores in scores_dict.items():
                debug_text = f"{p_name}: {p_scores['combined']:.2f}"
                cv2.putText(frame, debug_text, (x1, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 18
    
    # Debug mode indicator
    if app_state.debug_mode:
        cv2.putText(frame, "[DEBUG MODE]", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 4. Fallback: If target not found, track the largest person (closest)
    # This allows anyone to bring the gimbal back to the start
    if target_box is None and persons:
        # Find largest person by area
        best_area = 0
        best_idx = -1
        
        for i, (_, (x1, y1, x2, y2)) in enumerate(persons):
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_idx = i
        
        if best_idx != -1:
            # Use this person as target
            _, (x1, y1, x2, y2) = persons[best_idx]
            target_box = (x1, y1, x2, y2)
            
            # Visual indication for fallback (only if not already drawn)
            # We need to find if this person was already drawn as "Unknown"
            # The previous loop drew everyone, but we might want to highlight the tracked target
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3) # Yellow for fallback
            cv2.putText(frame, "Not Current Player", (x1, y1 - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 5. Control Gimbal if target found
    if target_box and app_state.tracker and app_state.tracker.connected:
        cx = (target_box[0] + target_box[2]) // 2
        cy = (target_box[1] + target_box[3]) // 2
        
        error_x = (app_state.tracker.frame_center_x - cx) / app_state.tracker.frame_center_x
        error_y = (app_state.tracker.frame_center_y - cy) / app_state.tracker.frame_center_y
        
        pan_speed = app_state.tracker._calculate_pan_speed(error_x, is_tennis=False)
        new_tilt_angle = app_state.tracker._calculate_tilt_angle(error_y)
        
        app_state.tracker.current_tilt_angle = new_tilt_angle
        app_state.tracker._send_command(pan_speed, new_tilt_angle)

    elif app_state.tracker and app_state.tracker.connected:
        # Target lost/not found this frame
        app_state.tracker._send_command(90, app_state.tracker.current_tilt_angle)
    
    # Emit player crop (always emit when target found, regardless of tracker)
    if target_box and app_state.current_player_crop is not None:
        emit_player_crop(app_state.current_player_crop, target_player_name)
    
    # Mark that player has been seen (for state flow control)
    if target_box:
        app_state.player_seen_after_count = True
    
    # 5. Check for ball (use correct class name: tennis_ball)
    # Only look for ball if player has been seen after last count
    all_objects = app_state.yolo.get_everything_attr(results)
    ball_found = any(cls.lower() == 'tennis_ball' for cls, _, _ in all_objects)
    
    if ball_found and app_state.player_seen_after_count:
        app_state.ball_visible_count += 1
        if app_state.ball_visible_count >= app_state.BALL_DETECT_THRESHOLD:
            gm.on_ball_detected()
            app_state.ball_visible_count = 0
            socketio.emit('tracking_status', {'target': 'tennis_ball', 'found': True})
            socketio.emit('game_state', gm.get_game_state_dict())
    else:
        app_state.ball_visible_count = 0
    
    return frame

def track_ball(frame, results):
    """Track the ball and detect strikes (including when all bottles knocked down)"""
    gm = app_state.game_manager
    
    # Track ball with gimbal
    if app_state.tracker:
        result = app_state.tracker.camera_track(frame, "tennis_ball", conf=app_state.conf, iou=app_state.iou)
        
        if result['found']:
            app_state.ball_lost_count = 0
            x1, y1, x2, y2 = result['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(frame, "Ball", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            app_state.ball_lost_count += 1
    
    # Check for bottles and player visibility
    bottles = get_standing_bottles(results)
    persons = app_state.yolo.get_person_crops(frame, results)
    player_visible = len(persons) > 0
    bottles_visible = len(bottles) > 0
    
    # Track player visibility state
    if player_visible:
        # Player is visible - reset timers and counts
        app_state.player_left_time = None
        app_state.bottles_visible_count = 0
        cv2.putText(frame, "Player in frame - waiting...", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    else:
        # Player has left frame - start timing
        if app_state.player_left_time is None:
            app_state.player_left_time = time.time()
        
        time_since_player_left = time.time() - app_state.player_left_time
        
        if bottles_visible:
            # Case 1: Bottles are visible, player is gone
            app_state.bottles_visible_count += 1
            
            if app_state.bottles_visible_count >= app_state.BOTTLE_DETECT_THRESHOLD:
                # Transition to counting state
                gm.on_bottles_visible()
                app_state.counting_start_time = time.time()
                app_state.player_left_time = None  # Reset for next throw
                bottle_label = app_state.yolo.bottle_label if app_state.yolo else 'bottle'
                socketio.emit('tracking_status', {'target': bottle_label, 'found': True})
                socketio.emit('game_state', gm.get_game_state_dict())
        else:
            # Case 2: No bottles visible, player is gone
            # This could be a strike (all bottles knocked down)
            app_state.bottles_visible_count = 0
            remaining = app_state.NO_BOTTLE_STRIKE_TIMEOUT - time_since_player_left
            
            if remaining > 0:
                # Still waiting - show countdown
                cv2.putText(frame, f"No bottles detected - Strike check in {remaining:.1f}s", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # 5 seconds passed with no bottles - it's a strike!
                print("STRIKE DETECTED: No bottles visible after throw!")
                app_state.player_left_time = None
                
                # Directly record the throw as a strike (0 standing bottles)
                gm.state = GameState.COUNTING
                result = gm.record_throw(0)  # 0 standing bottles = all knocked down
                
                socketio.emit('score_update', result)
                socketio.emit('game_state', gm.get_game_state_dict())
                
                # Reset tracking state
                app_state.ball_visible_count = 0
                app_state.ball_lost_count = 0
                app_state.bottles_visible_count = 0
                app_state.counting_start_time = None
                app_state.bottle_count_history = []
                app_state.player_seen_after_count = False  # Wait for player before next throw
                
                if result.get('game_over'):
                    socketio.emit('game_over', gm.get_game_state_dict())
    
    return frame

def track_bottles(frame, results):
    """Track bottles and start counting"""
    gm = app_state.game_manager
    
    # Track bottles with gimbal (use model-specific label)
    if app_state.tracker:
        bottle_label = app_state.yolo.bottle_label
        app_state.tracker.camera_track(frame, bottle_label, conf=app_state.conf, iou=app_state.iou)
    
    # # Draw bottles
    # bottles = get_standing_bottles(results)
    # for (x1, y1, x2, y2) in bottles:
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, "Standing", (x1, y1 - 10),
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Check countdown
    if app_state.counting_start_time:
        elapsed = time.time() - app_state.counting_start_time
        remaining = max(0, int(app_state.BOTTLE_COUNTING_TIMEOUT) - int(elapsed))
        
        socketio.emit('countdown', {'seconds': remaining})
        
        if elapsed >= app_state.BOTTLE_COUNTING_TIMEOUT:
            gm.state = GameState.COUNTING
            socketio.emit('countdown', {'seconds': 0})
    
    return frame

def count_bottles(frame, results):
    """Count standing bottles and record throw with optional temporal smoothing.
    Uses center crop of the frame for more accurate detection.
    """
    gm = app_state.game_manager
    
    # Crop center 2/3 of the frame for more focused detection
    h, w = frame.shape[:2]
    crop_x1 = w // 6        # 16.7% from left
    crop_x2 = w * 5 // 6    # 83.3% from left
    crop_y1 = h // 6        # 16.7% from top
    crop_y2 = h * 5 // 6    # 83.3% from top
    
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Run detection on cropped frame
    cropped_results = app_state.yolo.detect(cropped_frame, conf=app_state.conf, iou=app_state.iou)
    bottles = get_standing_bottles(cropped_results)
    raw_count = len(bottles)
    
    # Draw crop region on original frame (for visualization)
    cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 255, 0), 2)
    cv2.putText(frame, "Detection Zone", (crop_x1, crop_y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Apply temporal smoothing if enabled
    if app_state.temporal_smoothing_enabled:
        # Add current count to history
        app_state.bottle_count_history.append(raw_count)
        
        # Keep only the last N counts
        if len(app_state.bottle_count_history) > app_state.TEMPORAL_SMOOTHING_WINDOW:
            app_state.bottle_count_history.pop(0)
        
        # Use mode (most frequent value) for smoothed count
        from collections import Counter
        count_freq = Counter(app_state.bottle_count_history)
        smoothed_count = count_freq.most_common(1)[0][0]
        standing_count = smoothed_count
        
        # Display both raw and smoothed counts
        cv2.putText(frame, f"Raw: {raw_count} | Smoothed: {smoothed_count}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"[Temporal Smoothing ON]", (50, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        standing_count = raw_count
        # Display count (no smoothing)
        cv2.putText(frame, f"Standing: {standing_count}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    # Draw bottles on original frame (adjust coordinates from cropped to original)
    for (x1, y1, x2, y2) in bottles:
        # Offset coordinates back to original frame
        orig_x1 = x1 + crop_x1
        orig_y1 = y1 + crop_y1
        orig_x2 = x2 + crop_x1
        orig_y2 = y2 + crop_y1
        cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
    
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
    app_state.bottle_count_history = []  # Clear history for next throw
    app_state.player_left_time = None
    app_state.player_seen_after_count = False  # Wait for player before next throw
    
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
        # Resize crop maintaining aspect ratio
        h, w = crop.shape[:2]
        target_size = 200
        
        if h > target_size or w > target_size:
            if h > w:
                new_h = target_size
                new_w = int(w * (target_size / h))
            else:
                new_w = target_size
                new_h = int(h * (target_size / w))
            crop_resized = cv2.resize(crop, (new_w, new_h))
        else:
            crop_resized = crop
        
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
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    parser.add_argument('--debug_reid', action='store_true', help='Enable ReID debug overlay (shows all persons with scores)')
    
    args = parser.parse_args()
    
    # Set debug mode for ReID visualization
    app_state.debug_mode = args.debug_reid
    
    try:
        print("=" * 50)
        print("🎳 Bowling Scorer Web Application")
        print("=" * 50)
        print(f"Starting server on http://{args.host}:{args.port}")
        print(f"ReID Debug Mode: {'ON' if args.debug_reid else 'OFF'}")
        print("Open a browser to access the application")
        print("=" * 50)
        
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()
