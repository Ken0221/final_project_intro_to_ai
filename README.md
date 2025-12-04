# AI-Powered Automatic Scoring System for Toy Bowling

This project implements an automatic scoring system for a toy bowling game using computer vision. It utilizes **YOLOv8** for object detection (bottles and persons) and **MobileNetV2** for player re-identification (Re-ID).

## Features

*   **Real-time Bottle Detection**: Detects bowling bottles and determines their status (standing or fallen) based on aspect ratio.
*   **Automatic Scoring**: Counts the number of standing bottles in real-time.
*   **Player Identification**: Identifies players using a One-Shot Learning approach with MobileNetV2.
*   **Player Registration**: Allows registering new players on the fly using the camera.
*   **Visual Feedback**: Draws bounding boxes and labels for bottles and players on the video feed.

## Requirements

*   Python 3.8+
*   Webcam or video file source

## Installation

1.  Clone the repository (if applicable).
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: `requirements.txt` includes `opencv-python`, `ultralytics`, `torch`, and `torchvision`.*

## Usage

Run the main script to start the system:

```bash
python main.py --source 0
```

*   Replace `0` with your camera index or a video file path if needed.
*   The YOLOv8 nano model (`yolov8n.pt`) will be downloaded automatically on the first run.

### Controls

*   **`r`**: Enter **Registration Mode**.
    *   You will be prompted to enter a player name in the terminal.
    *   Stand in front of the camera to capture your reference image.
*   **`q`**: Quit the application.

## Project Structure

*   `main.py`: The main entry point. Handles video capture, YOLO inference, and display logic.
*   `player_reid.py`: Contains the `PlayerReID` class for feature extraction and player matching using MobileNetV2.
*   `check_cameras.py`: Utility script to list available camera sources.
*   `collect_data.py`: Script for collecting training/testing data (if needed).
*   `player_db/`: Directory where registered player feature vectors (`.npy` files) are stored.
*   `datasets/`: Directory for datasets (if applicable).

## How it Works

1.  **Bottle Detection**: YOLOv8 detects bottles. The system calculates the aspect ratio (height/width) of each bounding box. If the ratio is above a threshold (default 1.2), the bottle is considered "Standing"; otherwise, it is "Fallen".
2.  **Player Re-ID**: YOLOv8 detects persons. The detected person image is cropped and passed to a pre-trained MobileNetV2 (without the classifier) to extract a feature vector. This vector is compared with registered players in `player_db` using cosine similarity.
