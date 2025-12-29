from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        self.model_path = model_path
        
        # Check if CUDA is available before forcing it
        import torch
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            print(f"YOLO Detector initialized on device: {self.model.device}")
        else:
            print(f"YOLO Detector initialized on device: {self.model.device} (CPU)")
        
        # Initialize class IDs based on model type
        self._init_class_ids()

    def _init_class_ids(self):
        """
        Smart class ID detection based on model type.
        - merged_v2, merged_v3: Custom class IDs (person=2, bottle=1, tennisball=0)
        - COCO pretrained models: Standard COCO IDs
        """
        model_name = self.model_path.lower()
        
        # Check if using custom merged models
        if 'merged' in model_name:
            # Custom model class mapping
            self.CLASS_PERSON = 2
            self.CLASS_BOTTLE = 1
            self.CLASS_TENNISBALL = 0  # Direct tennis ball class in merged models
            self.TENNIS_BALL_CLASSES = {0}  # Only class 0 is tennis ball
            self.is_custom_model = True
            print(f"Using CUSTOM model class mapping: person={self.CLASS_PERSON}, bottle={self.CLASS_BOTTLE}, tennisball={self.CLASS_TENNISBALL}")
        else:
            # COCO pretrained model class mapping
            self.CLASS_PERSON = 0   # COCO: person is class 0
            self.CLASS_BOTTLE = 39  # COCO: bottle is class 39
            self.CLASS_TENNISBALL = 32  # COCO: sports ball
            # Tennis ball mapping - treat these COCO classes as tennis_ball
            # sports ball (32), frisbee (29), baseball glove (35), apple (47), orange (49)
            # self.TENNIS_BALL_CLASSES = {32, 29, 35, 47, 49}
            self.TENNIS_BALL_CLASSES = {32}
            self.is_custom_model = False
            print(f"Using COCO model class mapping: person={self.CLASS_PERSON}, bottle={self.CLASS_BOTTLE}")
        
        # Verify class names exist in model (for debugging)
        self._verify_classes()
        
        # Store actual label names for tracking
        self._init_label_names()
    
    def _init_label_names(self):
        """Initialize the actual label names used by the model for tracking."""
        if self.is_custom_model:
            # Merged models use 'plastic-bottle' as the class name
            self.bottle_label = self.model.names[self.CLASS_BOTTLE]
            self.person_label = self.model.names[self.CLASS_PERSON]
        else:
            # COCO models use 'bottle' and 'person'
            self.bottle_label = "bottle"
            self.person_label = "person"
        print(f"Tracking labels: person='{self.person_label}', bottle='{self.bottle_label}'")

    def _verify_classes(self):
        """Verify and print available classes in the model."""
        if hasattr(self.model, 'names'):
            class_names = self.model.names
            print(f"Model has {len(class_names)} classes")
            
            # Find person and bottle by name as fallback
            for cls_id, name in class_names.items():
                name_lower = name.lower()
                if 'person' in name_lower or 'human' in name_lower:
                    if self.CLASS_PERSON != cls_id:
                        print(f"  Note: Found 'person' at class {cls_id} (using {self.CLASS_PERSON})")
                if 'bottle' in name_lower:
                    if self.CLASS_BOTTLE != cls_id:
                        print(f"  Note: Found 'bottle' at class {cls_id} (using {self.CLASS_BOTTLE})")

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
                    # Ensure valid crop dimensions
                    if x2 > x1 and y2 > y1:
                        person_img = frame[y1:y2, x1:x2]
                        if person_img.size > 0:
                            persons.append((person_img, (x1, y1, x2, y2)))
        return persons

    def get_bottles(self, results):
        """
        Extract bottle bounding boxes.
        Returns a list of box coordinates.
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
                
                # Map certain classes to tennis_ball for consistent naming
                if cls_id in self.TENNIS_BALL_CLASSES:
                    cls_name = "tennis_ball"
                else:
                    cls_name = self.model.names[cls_id]
                
                attrs.append((cls_name, tuple(map(int, box.xyxy[0])), confidence))
        return attrs
