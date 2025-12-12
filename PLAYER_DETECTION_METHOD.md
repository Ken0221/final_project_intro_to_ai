# Player Detection / Re‑ID Method (Updated)

This document explains the updated **player detection / identification** method implemented in `player_reid.py`.  
Only the player Re‑ID component was changed; YOLO detection and the scoring/bottle logic remain the same.

---

## 1. Background and Problem

The original system follows the proposal:

1. Use YOLO to detect persons.
2. Crop each person and pass the crop into a pretrained backbone (MobileNet/ResNet/ConvNeXt).
3. Store one (or a few) feature vectors per player.
4. Identify a player by cosine similarity against stored vectors.

In practice, one‑shot accuracy was poor because the deep embedding alone was unstable under:
- background clutter (floor, lane, bottles)
- pose / scale changes
- partial occlusion
- lighting variation

These effects cause embeddings of different players to overlap.

---

## 2. Overview of the Updated Method

Each detected person is represented by a **hybrid appearance feature**:

**Hybrid feature = [Deep embedding] + [Clothing color histogram]**

Identification uses **cosine similarity on the hybrid feature** combined with your original
per‑player max/mean scoring.

Key upgrades:
1. **Torso‑focused crop**
2. **HSV clothing color descriptor**
3. **Light test‑time augmentation (TTA)**
4. **More reliable few‑shot registration target selection**

---

## 3. Step‑by‑Step Pipeline

### 3.1 Person detection (unchanged)

`YOLODetector.get_person_crops` returns crops and boxes for every COCO `person`.

### 3.2 Torso‑focused crop

Implemented in `PlayerReID._prepare_crop`.

Given YOLO’s full body crop, we re‑crop to emphasize clothing:
- If the crop looks full‑body (`height/width > 1.4`), keep roughly **12%–88%** vertical range.
- Otherwise keep **5%–95%**.
- Slightly trim left/right borders.

**Why:** clothing (torso region) is the most discriminative cue in your toy‑bowling setting.  
Reducing head/legs/background decreases feature drift.

### 3.3 Clothing color histogram

Implemented in `PlayerReID._compute_color_hist`.

- Convert crop to HSV.
- Compute a 2D histogram over **Hue (18 bins)** and **Saturation (8 bins)**.
- Normalize to unit length.

This yields a compact color signature (~144 dims) robust to minor lighting shifts.

**Why:** In your scenario, players often differ by shirt color/pattern.  
Deep ImageNet features are not trained specifically for clothing; explicit color helps separate close embeddings.

### 3.4 Deep embedding (backbone)

Backbone stays as your selectable pretrained network (`resnet50`, `resnet101`, etc.).

Implemented in `PlayerReID._deep_feature_from_pil`:
- Standard ImageNet preprocessing.
- Forward pass with classifier removed.
- L2‑normalize to unit length.

### 3.5 Hybrid feature assembly

Implemented in `PlayerReID.extract_features`:

`combined = concat(deep_feat, color_feat)`

Both parts are normalized before concatenation.

### 3.6 Registration (few‑shot)

The registration UI is now **multi‑angle first, name later** (matches the proposal’s few‑shot suggestion).

Main workflow (in `main.py`):

1. Press `r` to **capture one registration sample** for the *same player*.
   - YOLO detects persons; the largest person crop is buffered.
   - While collecting samples (after the first `r` and before `s`), all detected persons are displayed as `Unknown` to avoid confusing the on-screen IDs.
2. Change to a new angle/pose and press `r` again.
3. Repeat until you have **2–3 angles** (or more).
4. Press `s` to **finalize registration**.
5. Enter the player’s name **once** in the terminal.
6. All buffered crops are registered together via `PlayerReID.register_player_from_samples`.

Inside `PlayerReID.register_player_from_samples`:
- Each crop generates multiple hybrid features using the same augmentations as before:
  - original
  - color jitter
  - center crop
  - horizontal flip
  - small affin

These become a **multi‑sample prototype set** for that player.

**Why:** collecting 2–3 real angles *before naming* makes the prototype more complete and avoids repeated typing.

### 3.7 Identification (hybrid similarity)

Implemented in `PlayerReID.identify_player`.

1. Extract target hybrid feature **with TTA**:
   - original
   - horizontal flip  
   Average deep embeddings, then concatenate with color histogram.

2. For each stored sample:
   - split into deep/color parts if available
   - cosine similarity on deep
   - cosine similarity on color (if present)
   - fuse:

   `sim = 0.75 * sim_deep + 0.25 * sim_color`

3. Aggregate per player:

`combined_score = 0.7 * max(sim) + 0.3 * mean(sim)`

4. Use your margin‑based thresholding to decide Unknown vs matched player.

---

## 4. Why This Improves One‑Shot / Few‑Shot Detection

1. **Torso focus** increases signal‑to‑noise in both deep and color cues.
2. **Color histogram** encodes stable clothing properties that are often the main between‑player difference.
3. **TTA averaging** reduces single‑frame randomness (pose/blur/occlusion).
4. **Few‑shot prototypes** allow matching to the “best” sample while still using mean for robustness.

Net effect:
- **intra‑player similarity goes up** (same player stays clustered)
- **inter‑player similarity goes down** (different shirts separate better)

---

## 5. Practical Tuning Guide

### 5.1 Re‑register after update

Old `.npy` files only contain deep embeddings.  
To benefit from color features, delete and re‑register:

```bash
del player_db\*.npy
python main.py --source 0
```

### 5.2 Backbone choice

Accuracy ranking (slow → fast):
- `resnet101` best
- `resnet50` strong default
- `convnext_small` good balance
- `mobilenetv2` fastest, weakest

### 5.3 Threshold

`identify_player(..., threshold=0.55)`:
- Too many **Unknown** → try **0.50–0.53**
- Too many **mis‑IDs** → try **0.58–0.62**

### 5.4 Deep vs color weight

In `__init__`:

`self.deep_sim_weight = 0.75`

If players mainly differ by color, you can lower to **0.65**.  
If colors are similar but textures differ, raise to **0.85**.

### 5.5 Data capture tips

- Register each player **2–3 times** from different angles/lighting.
- Ensure the camera sees **upper body clearly**.
- Avoid excessive background motion during registration.

---

## 6. Limitations and Future Upgrades

- If two players wear very similar colors, the color cue helps less; deep cue dominates.
- Extreme lighting shifts can still distort HSV distribution.

Possible next improvements (still only player side):
- short‑term multi‑frame voting / tracking
- background segmentation before feature extraction
- lightweight fine‑tuning on your own player crops

---

If you want, I can add a tiny tracking/voting layer (without touching bottle logic) to stabilize IDs across frames.
