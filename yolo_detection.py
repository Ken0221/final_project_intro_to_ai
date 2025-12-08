from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLOv8 model
        # 'yolov8n.pt' is the nano version, faster but less accurate.
        # It will be downloaded automatically on first run.
        self.model = YOLO(model_path)
        
        # COCO class IDs
        self.CLASS_PERSON = 0
        self.CLASS_BOTTLE = 39

    def detect(self, frame, verbose=False):
        """
        Run inference on a frame.
        """
        return self.model(frame, verbose=verbose)

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
                    bottles.append(box)
        return bottles

    def get_everything_attr(self, results):
        """
        Extract all attributes from results.
        Returns a list of (attr, box_coords) tuples.
        box_coords = (x1, y1, x2, y2)
        """
        attrs = []
        for result in results:
            for box in result.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                attrs.append((cls_name, box.xyxy[0]))
        return attrs