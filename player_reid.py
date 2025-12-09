import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2


class PlayerReID:
    # Available backbone models
    AVAILABLE_MODELS = {
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
        'resnet101': (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
        'convnext_small': (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
        'mobilenetv2': (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1, 1280),
    }

    def __init__(self, db_path="player_db", backbone='resnet50'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PlayerReID using device: {self.device}")
        
        # Select backbone model
        if backbone not in self.AVAILABLE_MODELS:
            print(f"Unknown backbone '{backbone}', falling back to resnet50")
            backbone = 'resnet50'
        
        self.backbone_name = backbone
        model_fn, weights, self.feature_dim = self.AVAILABLE_MODELS[backbone]
        print(f"Using backbone: {backbone} (feature dim: {self.feature_dim})")
        
        # Load pre-trained model
        self.model = model_fn(weights=weights)
        
        # Remove the classifier to get feature vectors
        if backbone.startswith('resnet'):
            self.model.fc = nn.Identity()
        elif backbone == 'convnext_small':
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],  # Keep LayerNorm
                nn.Flatten(1),
            )
        elif backbone == 'mobilenetv2':
            self.model.classifier = nn.Identity()
        
        self.model.to(self.device)
        self.model.eval()

        # Standard transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # Augmentation transforms for registration
        # Enhanced for "Fixed Environment" usage:
        # We removed ColorJitter (lighting is constant) and Affine (too much noise).
        # We keep Crop (distance variation) and Flip (angle variation).
        self.augment_transforms = [
            # transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ]),
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),  # Always flip
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            # transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ]),
        ]

        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        # Changed to store multiple features per player: {name: [feature_vectors]}
        self.known_players = {}
        self.debug_mode = False
        self.load_db()

    def extract_features(self, img_bgr, use_augment=False):
        """
        Extract feature vector(s) from a BGR image (numpy array).
        If use_augment=True, returns multiple augmented features.
        """
        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        if use_augment:
            # Extract features with augmentation for registration
            all_features = []
            
            # Original image
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(img_tensor)
            all_features.append(features.cpu().numpy().flatten())
            
            # Augmented versions
            for aug_transform in self.augment_transforms:
                img_tensor = aug_transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model(img_tensor)
                all_features.append(features.cpu().numpy().flatten())
            
            return all_features
        else:
            # Standard extraction
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(img_tensor)
            return features.cpu().numpy().flatten()

    def register_player(self, name, img_bgr, capture_multiple=True, cap=None, model=None):
        """
        Register a new player with reference images.
        If capture_multiple is True and cap/model provided, captures multiple frames.
        Each frame generates multiple augmented features.
        """
        all_features = []
        frames_to_process = [img_bgr]
        '''
        # Capture additional frames if possible
        if capture_multiple and cap is not None and model is not None:
            import time
            print("Capturing additional frames for better registration...")
            for i in range(2):  # Capture 2 more frames
                time.sleep(0.5)  # Wait a bit between frames
                ret, frame = cap.read()
                if ret:
                    # Detect person in this frame
                    results = model(frame, verbose=False)
                    # Find largest person in this frame
                    best_person_crop = None
                    max_area = 0
                    
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls[0]) == 0:  # Person
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                area = (x2 - x1) * (y2 - y1)
                                if area > max_area:
                                    max_area = area
                                    person_crop = frame[y1:y2, x1:x2]
                                    if person_crop.size > 0:
                                        best_person_crop = person_crop
                    
                    if best_person_crop is not None:
                        frames_to_process.append(best_person_crop)
        '''
        # Extract features from all frames with augmentation
        for frame in frames_to_process:
            features_list = self.extract_features(frame, use_augment=True)
            all_features.extend(features_list)
        
        # Add to existing player or create new
        if name in self.known_players:
            self.known_players[name].extend(all_features)
            print(f"Added {len(all_features)} samples to existing player: {name}")
        else:
            self.known_players[name] = all_features
            print(f"Registered new player: {name} with {len(all_features)} samples")
        
        # Save to disk (as multiple features)
        features_array = np.array(self.known_players[name])
        np.save(os.path.join(self.db_path, f"{name}.npy"), features_array)
        
        return len(all_features)

    def identify_player(self, img_bgr, threshold=0.55):
        """
        Identify player from image using enhanced similarity calculation.
        Returns (name, score, all_scores) or (None, score, all_scores).
        all_scores is a dict for debug mode.
        """
        if not self.known_players:
            return None, 0.0, {}

        target_features = self.extract_features(img_bgr)
        
        scores_dict = {}  # For debug mode
        best_combined_score = -1.0
        second_best_score = -1.0
        best_match = None

        for name, features_list in self.known_players.items():
            # Calculate similarity with all registered features for this player
            similarities = []
            for features in features_list:
                score = np.dot(target_features, features) / (np.linalg.norm(target_features) * np.linalg.norm(features))
                similarities.append(score)
            
            # Combined score: 70% max similarity + 30% mean similarity
            max_sim = max(similarities)
            mean_sim = np.mean(similarities)
            combined_score = 0.7 * max_sim + 0.3 * mean_sim
            
            scores_dict[name] = {
                'max': max_sim,
                'mean': mean_sim,
                'combined': combined_score,
                'samples': len(features_list)
            }
            
            if combined_score > best_combined_score:
                second_best_score = best_combined_score
                best_combined_score = combined_score
                best_match = name
            elif combined_score > second_best_score:
                second_best_score = combined_score

        # Margin-based decision: if two players are too close, raise threshold
        margin = best_combined_score - second_best_score if second_best_score > 0 else 1.0
        effective_threshold = threshold
        if margin < 0.05 and len(self.known_players) > 1:
            effective_threshold = max(threshold, 0.65)
            if self.debug_mode:
                print(f"[Debug] Low margin ({margin:.3f}), using higher threshold: {effective_threshold:.2f}")

        if best_combined_score > effective_threshold:
            return best_match, best_combined_score, scores_dict
        else:
            return None, best_combined_score, scores_dict

    def delete_player(self, name):
        """Delete a registered player."""
        if name in self.known_players:
            del self.known_players[name]
            filepath = os.path.join(self.db_path, f"{name}.npy")
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Deleted player: {name}")
            return True
        else:
            print(f"Player '{name}' not found.")
            return False

    def list_players(self):
        """List all registered players with their sample counts."""
        if not self.known_players:
            print("No players registered.")
            return {}
        
        print("\n=== Registered Players ===")
        player_info = {}
        for name, features_list in self.known_players.items():
            sample_count = len(features_list)
            player_info[name] = sample_count
            print(f"  {name}: {sample_count} samples")
        print("==========================\n")
        return player_info

    def set_debug_mode(self, enabled):
        """Enable or disable debug mode."""
        self.debug_mode = enabled
        print(f"Debug mode: {'ON' if enabled else 'OFF'}")

    def load_db(self):
        """Load registered players from disk."""
        print("Loading player database...")
        for filename in os.listdir(self.db_path):
            if filename.endswith(".npy"):
                name = os.path.splitext(filename)[0]
                features = np.load(os.path.join(self.db_path, filename))
                # Handle both old format (1D array) and new format (2D array)
                if features.ndim == 1:
                    # Old format: single feature vector
                    self.known_players[name] = [features]
                else:
                    # New format: multiple feature vectors
                    self.known_players[name] = list(features)
                print(f"Loaded {name} ({len(self.known_players[name])} samples)")
