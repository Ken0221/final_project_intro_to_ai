import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class PlayerReID:
    def __init__(self, db_path="player_db"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PlayerReID using device: {self.device}")
        
        # Load pre-trained MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Remove the classifier (last layer) to get feature vectors
        # MobileNetV2 structure: features -> classifier
        # We want the output of 'features' + global average pooling
        self.model.classifier = nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.known_players = {} # {name: feature_vector}
        self.load_db()

    def extract_features(self, img_bgr):
        """Extract feature vector from a BGR image (numpy array)."""
        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocess
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            # MobileNetV2 features output is [1, 1280, 7, 7] before pooling if we just took .features
            # But we replaced .classifier with Identity, so it goes through the pooling in the original forward?
            # Wait, MobileNetV2 forward is: features -> nn.functional.adaptive_avg_pool2d((1, 1)) -> flatten -> classifier
            # If we replace classifier with Identity, we get the flattened vector.
            # Let's verify MobileNetV2 source or behavior. 
            # Actually, standard torchvision mobilenet_v2 forward does:
            # x = self.features(x)
            # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            # x = torch.flatten(x, 1)
            # x = self.classifier(x)
            # So Identity classifier gives us the 1280-dim vector. Correct.
            
        return features.cpu().numpy().flatten()

    def register_player(self, name, img_bgr):
        """Register a new player with a reference image."""
        features = self.extract_features(img_bgr)
        self.known_players[name] = features
        
        # Save to disk
        np.save(os.path.join(self.db_path, f"{name}.npy"), features)
        print(f"Registered player: {name}")

    def identify_player(self, img_bgr, threshold=0.5):
        """Identify player from image. Returns (name, score) or (None, score)."""
        if not self.known_players:
            return None, 0.0

        target_features = self.extract_features(img_bgr)
        
        best_score = -1.0
        best_match = None

        for name, features in self.known_players.items():
            # Cosine similarity
            score = np.dot(target_features, features) / (np.linalg.norm(target_features) * np.linalg.norm(features))
            
            if score > best_score:
                best_score = score
                best_match = name

        if best_score > threshold:
            return best_match, best_score
        else:
            return None, best_score

    def load_db(self):
        """Load registered players from disk."""
        print("Loading player database...")
        for filename in os.listdir(self.db_path):
            if filename.endswith(".npy"):
                name = os.path.splitext(filename)[0]
                features = np.load(os.path.join(self.db_path, filename))
                self.known_players[name] = features
                print(f"Loaded {name}")

import cv2 # Need cv2 for color conversion inside extract_features
