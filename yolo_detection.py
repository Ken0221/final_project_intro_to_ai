from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLOv8 model
        # 'yolov8n.pt' is the nano version, faster but less accurate.
        # It will be downloaded automatically on first run.
        # Use GPU 0 if available, explicitly
        self.model = YOLO(model_path)
        
        # Check if CUDA is available before forcing it
        import torch
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            print(f"YOLO Detector initialized on device: {self.model.device}")
        else:
            print(f"YOLO Detector initialized on device: {self.model.device} (CPU)")
        
        # COCO class IDs
        self.CLASS_PERSON = 2
        self.CLASS_BOTTLE = 1

        # Tennis ball mapping - treat these COCO classes as tennis_ball
        # sports ball (32), frisbee (29), baseball glove (35), apple (47), orange (49)
        self.TENNIS_BALL_CLASSES = {32, 29, 35, 47, 49}

    def detect(self, frame, conf=0.25, iou=0.7, verbose=False):
        
        """
        Run inference on a frame.
        """
        return self.model(frame, conf=conf, iou=iou, verbose=verbose)

    def get_person_crops(self, frame, results):
        """
        Extract person crops and their bounding boxes from results.
        Returns a list of (crop, box_coords) tuples.
        box_coords = (x1, y1, x2, y2)
        """
        persons = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.CLASS_PERSON:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Crop person
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size > 0:
                        persons.append((person_img, (x1, y1, x2, y2)))
        return persons

    def get_bottles(self, results):
        """
        Extract bottle bounding boxes.
        Returns a list of box objects.
        """
        bottles = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.CLASS_BOTTLE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bottles.append((x1, y1, x2, y2))
        return bottles

    def get_everything_attr(self, results):
        """
        Extract all attributes from results.
        Returns a list of (attr, box_coords, confidence) tuples.
        box_coords = (x1, y1, x2, y2)
        confidence = detection confidence score (0.0 - 1.0)
        """
        attrs = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # Map certain classes to tennis_ball
                if cls_id in self.TENNIS_BALL_CLASSES:
                    cls_name = "tennis_ball"
                else:
                    cls_name = self.model.names[cls_id]
                # cls_name = self.model.names[cls_id]
                attrs.append((cls_name, box.xyxy[0], confidence))
        return attrs