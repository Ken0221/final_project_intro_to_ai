# AI-Powered Automatic Scoring System for Toy Bowling

This project implements an automatic scoring system for a toy bowling game using computer vision. It utilizes **YOLOv8** for object detection (bottles and persons) and **ResNet50** (or other backbones) for player re-identification (Re-ID).

## Features

### Core Functionality
- **Real-time Bottle Detection**: Detects bowling bottles and determines their status (standing or fallen) based on aspect ratio.
- **Automatic Scoring**: Counts the number of standing bottles in real-time.
- **Advanced Player Identification**: Identifies players using deep learning with multiple backbone options.
- **Visual Feedback**: Draws bounding boxes and labels for bottles and players on the video feed.

### Player Re-ID Improvements
- **Multiple Backbone Models**: Choose from ResNet50 (default), ResNet101, ConvNeXt-Small, or MobileNetV2
- **Multi-Sample Registration**: Each registration captures multiple frames with data augmentation (up to 15 samples per registration)
- **Enhanced Similarity Calculation**: Combines max and mean similarity scores with margin-based thresholding
- **Incremental Registration**: Register the same player multiple times to improve accuracy
- **Debug Mode**: View detailed similarity scores for all registered players

## Requirements

- Python 3.8+
- Webcam or video file source
- CUDA-capable GPU (optional, for faster inference)

## Installation

1. Clone the repository (if applicable).
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: `requirements.txt` includes `opencv-python`, `ultralytics`, `torch`, `torchvision`, and other dependencies.*

## Dataset

* Plastic bottles
```
curl -L "https://universe.roboflow.com/ds/N4Y7qdOvNp?key=iTf4N8oQWT" > plastic_bottles.zip; unzip plastic_bottles.zip; rm plastic_bottles.zip
```
* Tennis balls
```
curl -L "https://universe.roboflow.com/ds/CMg9AG5skG?key=2PTgYv0CPu" > tennis_balls.zip; unzip tennis_balls.zip; rm tennis_balls.zip
```

## Usage

### Basic Usage

Run the main script to start the system:

```bash
python main.py --source 0
```

- Replace `0` with your camera index or a video file path if needed.
- The YOLOv8 nano model (`yolov8n.pt`) will be downloaded automatically on the first run.

### Advanced Options

```bash
# Use custom resolution
python main.py --source 0 --resolution 1080    # 1920x1080 (Default)
python main.py --source 0 --resolution 2160    # 4K (3840x2160)
python main.py --source 0 --resolution 720     # 1280x720

# Use different detection models (supports RT-DETR)
python main.py --source 0 --model yolov8m      # Medium model
python main.py --source 0 --model rtdetr-l     # RT-DETR Large model

# Show FPS and debug info
python main.py --source 0 --show_everything 1

# Use different backbone models for Re-ID
python main.py --source 0 --backbone resnet50      # Default, best accuracy
python main.py --source 0 --backbone resnet101     # Higher accuracy, slower
python main.py --source 0 --backbone convnext_small # Good balance
python main.py --source 0 --backbone mobilenetv2   # Fastest, lower accuracy

# Use camera presets
python main.py --source default  # Camera 0
python main.py --source mac      # Camera 1
python main.py --source ivcam    # Camera 2

# Use video file
python main.py --source path/to/video.mp4
```

### Keyboard Controls

| Key | Function |
|-----|----------|
| `r` | **Capture Registration Sample** - Capture one angle/pose for the same player (press multiple times) |
| `s` | **Save Registration** - After capturing 2–3 angles with `r`, enter player name once and save all samples |
| `d` | **Debug Mode** - Toggle display of all similarity scores for each detected person |
| `l` | **List Players** - Show all registered players and their sample counts |
| `c` | **Clear/Delete Player** - Remove a registered player from the database |
| `q` | **Quit** - Exit the application |

### Registration Workflow

1. Press `r` to capture a registration sample for a player
2. Change angle/pose and press `r` again (repeat 2–3 times)
   - While you are collecting samples (after the first `r` and before `s`), all detected persons are displayed as **Unknown**
3. Press `s` to finalize registration
4. Enter the player's name once in the terminal
5. The system will:
   - Use all captured angles
   - Generate 5 augmented features per angle (original + 4 augmentations)
   - Store a multi-sample prototype per player

## Project Structure

- `main.py`: The main entry point. Handles video capture, YOLO inference, and display logic.
- `player_reid.py`: Contains the `PlayerReID` class for feature extraction and player matching with multiple backbone support.
- `check_cameras.py`: Utility script to list available camera sources.
- `collect_data.py`: Script for collecting training/testing data (if needed).
- `player_db/`: Directory where registered player feature vectors (`.npy` files) are stored.
- `requirements.txt`: Python package dependencies.

## How it Works

### Bottle Detection
YOLOv8 detects bottles in each frame. The system calculates the aspect ratio (height/width) of each bounding box. If the ratio is above a threshold (default 1.2), the bottle is considered "Standing"; otherwise, it is "Fallen".

### Player Re-Identification

1. **Detection**: YOLOv8 detects persons in the frame
2. **Feature Extraction**: The detected person image is cropped and passed to a deep learning backbone (ResNet50 by default) to extract a high-dimensional feature vector (2048-dim for ResNet50)
3. **Matching**: The feature vector is compared with all registered players using:
   - **Cosine similarity** for each stored sample
   - **Combined score**: `0.7 × max_similarity + 0.3 × mean_similarity`
   - **Margin-based thresholding**: If two players have similar scores (margin < 0.05), the threshold is raised to 0.65
4. **Identification**: The player with the highest combined score above the threshold is identified

### Data Augmentation (Registration)

During registration, each captured frame is augmented using:
- Original image
- Color jitter (brightness/contrast variation)
- Center crop with resize
- Horizontal flip
- Random affine transformation (rotation + translation)

This creates robust feature representations that work under different lighting and pose conditions.

## Performance Tips

- **GPU Acceleration**: The system automatically uses CUDA if available
- **Model Selection**: 
  - Use `resnet50` for best accuracy
  - Use `mobilenetv2` if running on CPU or need faster inference
- **Registration**: Register each player 2-3 times from different angles for best results
- **Lighting**: Ensure consistent lighting between registration and identification

## Troubleshooting

### Player Not Recognized
- Register the player multiple times (press `r` again with the same name)
- Ensure good lighting conditions
- Try using a higher-capacity backbone (`--backbone resnet101`)

### Slow Performance
- Switch to a lighter backbone (`--backbone mobilenetv2`)
- Ensure CUDA is properly installed for GPU acceleration
- Reduce camera resolution if possible

### Wrong Player Identified
- Enable debug mode (`d` key) to see similarity scores
- Delete incorrect registrations (`c` key)
- Re-register players with more samples

## Migration from Old Version

If you have existing player data from the old MobileNetV2 version:
- The new system is **not compatible** with old `.npy` files (different feature dimensions)
- Recommended: Delete `player_db/*.npy` and re-register all players
- The system will automatically convert old single-sample files to multi-sample format if dimensions match

## License

[Specify your license here]

## Acknowledgments

- YOLOv8 by Ultralytics
- PyTorch and torchvision for deep learning models
- OpenCV for computer vision operations
