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

    # Color histogram settings for clothing-aware Re-ID
    COLOR_H_BINS = 18  # Hue bins (0-180 in OpenCV HSV)
    COLOR_S_BINS = 8   # Saturation bins (0-256)

    def __init__(self, db_path="player_db", backbone='resnet50'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PlayerReID using device: {self.device}")
        
        # Select backbone model
        if backbone not in self.AVAILABLE_MODELS:
            print(f"Unknown backbone '{backbone}', falling back to resnet50")
            backbone = 'resnet50'
        
        self.backbone_name = backbone
        model_fn, weights, self.feature_dim = self.AVAILABLE_MODELS[backbone]
        self.deep_dim = self.feature_dim
        self.color_dim = self.COLOR_H_BINS * self.COLOR_S_BINS
        # Weight for combining deep and color similarity
        self.deep_sim_weight = 0.75
        print(
            f"Using backbone: {backbone} "
            f"(deep dim: {self.deep_dim}, color dim: {self.color_dim})"
        )
        
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
        self.augment_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
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
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        ]

        # Test-time augmentation for inference (mild only)
        self.tta_transforms = [
            self.transform,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]),
        ]

        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        # Changed to store multiple features per player: {name: [feature_vectors]}
        self.known_players = {}
        self.debug_mode = False
        self.load_db()

    def _prepare_crop(self, img_bgr):
        """
        Lightly post-process a person crop:
        - Focus on torso region (clothing)
        - Avoid extreme small crops
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        if h < 60 or w < 30:
            return img_bgr

        # If full-body aspect ratio, keep torso mostly
        if h / max(w, 1) > 1.4:
            y1 = int(h * 0.12)
            y2 = int(h * 0.88)
        else:
            y1 = int(h * 0.05)
            y2 = int(h * 0.95)
        x1 = int(w * 0.05)
        x2 = int(w * 0.95)

        torso = img_bgr[y1:y2, x1:x2]
        return torso if torso.size > 0 else img_bgr

    def _compute_color_hist(self, img_bgr):
        """
        Compute a normalized HSV (H,S) histogram as a compact clothing descriptor.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [self.COLOR_H_BINS, self.COLOR_S_BINS],
            [0, 180, 0, 256]
        )
        hist = cv2.normalize(hist, None).flatten().astype(np.float32)
        norm = np.linalg.norm(hist) + 1e-12
        return hist / norm

    def _deep_feature_from_pil(self, pil_img, transform):
        img_tensor = transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        feat = features.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm

    def _split_feature(self, feat):
        """
        Split stored feature into deep/color parts.
        - deep only: len == deep_dim
        - deep+color: len == deep_dim + color_dim
        Any other length is treated as deep-only (truncate/pad).
        """
        feat = np.asarray(feat).flatten()
        if feat.size == self.deep_dim:
            deep = feat
            color = None
        elif feat.size == self.deep_dim + self.color_dim:
            deep = feat[:self.deep_dim]
            color = feat[self.deep_dim:]
        else:
            # Likely from a different backbone; fall back to deep-only safely
            if feat.size >= self.deep_dim:
                deep = feat[:self.deep_dim]
            else:
                deep = np.pad(feat, (0, self.deep_dim - feat.size))
            color = None
        return deep, color

    def extract_features(self, img_bgr, use_augment=False, tta=False):
        """
        Extract a combined appearance feature from a BGR crop.
        - Deep backbone embedding (instance-level)
        - HSV clothing histogram (color-level)
        If use_augment=True, returns multiple combined features for registration.
        If tta=True, averages a small set of test-time augmentations.
        """
        img_bgr = self._prepare_crop(img_bgr)
        if img_bgr is None or img_bgr.size == 0:
            return [] if use_augment else np.zeros(self.deep_dim + self.color_dim, dtype=np.float32)

        color_feat = self._compute_color_hist(img_bgr)

        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        if use_augment:
            # Extract deep features with augmentation for registration
            all_features = []

            # Original + augmented deep features
            deep_features = [self._deep_feature_from_pil(pil_img, self.transform)]
            for aug_transform in self.augment_transforms:
                deep_features.append(self._deep_feature_from_pil(pil_img, aug_transform))

            for deep_feat in deep_features:
                combined = np.concatenate([deep_feat, color_feat]).astype(np.float32)
                all_features.append(combined)

            return all_features
        else:
            # Standard extraction with optional test-time augmentation
            if tta:
                deep_list = [self._deep_feature_from_pil(pil_img, t) for t in self.tta_transforms]
                deep_feat = np.mean(deep_list, axis=0)
                deep_feat = deep_feat / (np.linalg.norm(deep_feat) + 1e-12)
            else:
                deep_feat = self._deep_feature_from_pil(pil_img, self.transform)

            return np.concatenate([deep_feat, color_feat]).astype(np.float32)

    def register_player_from_samples(self, name, frames_bgr):
        """
        Register (or extend) a player using a list of BGR person crops.
        Each crop generates multiple augmented hybrid features.
        """
        all_features = []
        for frame in frames_bgr:
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

    def register_player(self, name, img_bgr, capture_multiple=True, cap=None, model=None):
        """
        Register a new player with reference images.
        If capture_multiple is True and cap/model provided, captures multiple frames.
        Each frame generates multiple augmented features.
        """
        frames_to_process = [img_bgr]
        
        # Capture additional frames if possible
        if capture_multiple and cap is not None and model is not None:
            import time
            print("Capturing additional frames for better registration...")
            for i in range(2):  # Capture 2 more frames
                time.sleep(0.5)  # Wait a bit between frames
                ret, frame = cap.read()
                if ret:
                    # Detect persons and keep the largest one (registration target)
                    results = model(frame, verbose=False)
                    best_box = None
                    best_area = 0
                    for result in results:
                        for box in result.boxes:
                            if int(box.cls[0]) == 0:  # Person
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                area = max(0, x2 - x1) * max(0, y2 - y1)
                                if area > best_area:
                                    best_area = area
                                    best_box = (x1, y1, x2, y2)
                    if best_box is not None:
                        x1, y1, x2, y2 = best_box
                        person_crop = frame[y1:y2, x1:x2]
                        if person_crop.size > 0:
                            frames_to_process.append(person_crop)

        return self.register_player_from_samples(name, frames_to_process)

    def identify_player(self, img_bgr, threshold=0.55):
        """
        Identify player from image using enhanced similarity calculation.
        Returns (name, score, all_scores) or (None, score, all_scores).
        all_scores is a dict for debug mode.
        """
        if not self.known_players:
            return None, 0.0, {}

        target_features = self.extract_features(img_bgr, tta=True)
        target_deep, target_color = self._split_feature(target_features)
        target_deep_norm = np.linalg.norm(target_deep) + 1e-12
        target_color_norm = np.linalg.norm(target_color) + 1e-12 if target_color is not None else 0.0
        
        scores_dict = {}  # For debug mode
        best_combined_score = -1.0
        second_best_score = -1.0
        best_match = None

        for name, features_list in self.known_players.items():
            # Calculate similarity with all registered features for this player
            similarities = []
            for features in features_list:
                deep_feat, color_feat = self._split_feature(features)
                deep_norm = np.linalg.norm(deep_feat) + 1e-12
                deep_sim = float(np.dot(target_deep, deep_feat) / (target_deep_norm * deep_norm))

                if color_feat is not None and target_color is not None and target_color_norm > 0:
                    color_norm = np.linalg.norm(color_feat) + 1e-12
                    color_sim = float(np.dot(target_color, color_feat) / (target_color_norm * color_norm))
                    score = self.deep_sim_weight * deep_sim + (1.0 - self.deep_sim_weight) * color_sim
                else:
                    score = deep_sim

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
                # Warn if feature dims mismatch current backbone
                sample_len = self.known_players[name][0].size
                if sample_len not in (self.deep_dim, self.deep_dim + self.color_dim):
                    print(
                        f"Warning: {name} feature dim {sample_len} "
                        f"does not match current backbone (deep {self.deep_dim}). "
                        "Will fallback to deep-only/truncated matching."
                    )
                print(f"Loaded {name} ({len(self.known_players[name])} samples)")
