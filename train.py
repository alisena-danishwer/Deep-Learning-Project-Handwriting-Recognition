"""The method to run train.py
python train.py --train train --model_out model.keras --labels_out labels.json
"""
import os, json, random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import argparse

# ---------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default="train", help="Input directory containing training images")
parser.add_argument("--model_out", type=str, default="model.h5", help="Path to save the trained model")
parser.add_argument("--labels_out", type=str, default="labels.json", help="Path to save the label mapping")
parser.add_argument("--char_h", type=int, default=64, help="Target height for resized characters")
parser.add_argument("--char_w", type=int, default=64, help="Target width for resized characters")
parser.add_argument("--min_char_width", type=int, default=6, help="Minimum width in pixels to consider a segment a valid character")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--val_fraction", type=float, default=0.10, help="Percentage of data to use for validation")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ---------------------------------------------------------
# Reproducibility Setup
# ---------------------------------------------------------
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def list_images(folder):
    """Recursively finds all images with standard extensions in the directory."""
    p = Path(folder)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted([str(fp) for fp in p.rglob("*") if fp.suffix.lower() in exts])

def label_from_filename(fp):
    """
    Extracts class label from filename. 
    Assumes the first 2 characters of the filename represent the class.
    Example: 'A1_sample.jpg' -> Label 'A1'
    """
    return Path(fp).name[:2]

def to_grayscale(img):
    """Safely converts BGR to Grayscale if input is not None."""
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# ---------------------------------------------------------
# Segmentation Logic (Computer Vision)
# ---------------------------------------------------------
def segment_lines(gray):
    """
    Splits an image into lines of text using Horizontal Projection Profiles.
    It sums pixels horizontally; rows with pixels are text, empty rows are gaps.
    """
    h = gray.shape[0]
    # 1. Binarize: Invert so text is white (255) and background is black (0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Morphological Closing: Connects horizontal gaps to make lines solid blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 3. Projection: Sum pixels along the width (axis 1)
    proj = np.sum(closed, axis=1)
    
    if proj.max() == 0: return [(0, h)] # Return full image if it's blank
    
    # Define a threshold to filter noise (3% of max density)
    thresh = max(1, int(0.03 * proj.max()))
    
    lines = []
    in_line = False
    start = 0
    
    # 4. Iterate through the projection to find start/end indices of lines
    for y, v in enumerate(proj):
        if v > thresh and not in_line:
            in_line = True
            start = y
        elif v <= thresh and in_line:
            end = y
            in_line = False
            # Only keep line if it is tall enough (>=6 pixels)
            if end - start >= 6:
                # Add small padding (+/- 2 pixels)
                lines.append((max(0, start - 2), min(h, end + 2)))
    
    if in_line:
        lines.append((start, h))
    return lines

def segment_words_from_line(line_img):
    """
    Splits a line of text into words using dilation to merge characters.
    """
    gray = to_grayscale(line_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilation: Stretch white pixels horizontally (15x3 kernel) to merge letters into word blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(th, kernel, iterations=1)
    
    # Find bounding boxes of these merged blobs
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 8 or h < 8: continue # Ignore noise
        bboxes.append((x, y, w, h))
        
    # Sort words left-to-right based on x-coordinate
    bboxes = sorted(bboxes, key=lambda b: b[0])
    return [line_img[y:y+h, x:x+w] for (x, y, w, h) in bboxes]

def segment_chars_from_word(word_img, min_char_width=4):
    """
    Splits a word into characters using Vertical Projection Profiles.
    """
    gray = to_grayscale(word_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Sum pixels along the height (axis 0) to find vertical gaps between letters
    cols = np.sum(th, axis=0)
    thresh = max(1, int(0.05 * cols.max()))
    
    # Identify columns that are separators (gaps)
    separators = cols <= thresh
    chars = []
    in_char = False
    start = 0
    
    for i, val in enumerate(separators):
        if (not val) and (not in_char):
            # Gap ended, character started
            in_char = True
            start = i
        elif val and in_char:
            # Character ended, gap started
            end = i
            in_char = False
            if end - start >= min_char_width:
                chars.append(word_img[:, start:end])
                
    # Handle case where character goes to the very edge of image
    if in_char:
        end = len(separators)
        if end - start >= min_char_width:
            chars.append(word_img[:, start:end])
    return chars

def resize_and_normalize_char(ch_img, target_h, target_w):
    """
    Resizes image to target size while maintaining aspect ratio, then pads.
    Final output is normalized float32 [0, 1].
    """
    if len(ch_img.shape) == 2:
        ch = cv2.cvtColor(ch_img, cv2.COLOR_GRAY2RGB)
    else:
        ch = ch_img.copy()
        
    h, w = ch.shape[:2]
    # Calculate scale to fit within the box without distortion
    scale = min(max(1e-6, target_w/w), max(1e-6, target_h/h))
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    
    # Resize
    resized = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Center the resized image on a white background (pad)
    pad_left = (target_w - nw) // 2
    pad_top = (target_h - nh) // 2
    padded = 255 * np.ones((target_h, target_w, 3), dtype=np.uint8)
    padded[pad_top:pad_top+nh, pad_left:pad_left+nw, :] = resized
    
    # Normalize pixel values to 0.0 - 1.0
    return padded.astype(np.float32) / 255.0

# ---------------------------------------------------------
# Augmentation Setup
# ---------------------------------------------------------
# Define a Keras Sequential model strictly for data augmentation
augmentation_preseg = tf.keras.Sequential([
    layers.RandomRotation(0.20),         # Rotate +/- 20%
    layers.RandomTranslation(0.10, 0.10), # Shift +/- 10%
    layers.RandomZoom(0.15, 0.15),       # Zoom +/- 15%
    layers.RandomContrast(0.20),
    layers.GaussianNoise(0.02),          # Add grain
], name="strong_augmentation_preseg")

@tf.function
def _augment_tensor(x):
    """Wrapper to run augmentation inside TF graph."""
    return augmentation_preseg(x)

def augment_image_numpy(img_uint8):
    """
    Applies TF augmentation to a Numpy image.
    Converts uint8 -> float32 -> augment -> uint8.
    """
    img = img_uint8.astype(np.float32) / 255.0
    t = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
    aug = _augment_tensor(t)
    aug_np = aug[0].numpy()
    aug_uint8 = np.clip(aug_np * 255.0, 0, 255).astype(np.uint8)
    return aug_uint8

# ---------------------------------------------------------
# Data Loading & Processing Pipeline
# ---------------------------------------------------------
train_files = list_images(args.train)
if len(train_files) == 0: raise SystemExit("No training images found")

# Create label map (Class Name -> Integer Index)
labels = sorted({label_from_filename(f) for f in train_files})
label_to_index = {lab: i for i, lab in enumerate(labels)}

X, y = [], []

print(f"Found {len(train_files)} images. Starting processing...")
for fp in tqdm(train_files, desc="Processing train images"):
    img = cv2.imread(fp)
    if img is None: continue
    H, W = img.shape[:2]
    
    # HEURISTIC: Split the raw image into 3 vertical strips.
    # Assumes input data format is a wide strip or page that needs partitioning.
    third = W // 3
    patches = [img[:, :third], img[:, third:2*third], img[:, 2*third:]]
    
    # Augmentation Strategy:
    # For every 1 original patch, generate 1 augmented version.
    enhanced_patches = []
    for p in patches:
        enhanced_patches.append(p) # Add original
        try:
            aug = augment_image_numpy(p)
            enhanced_patches.append(aug) # Add augmented
        except Exception:
            # Fallback if augmentation fails (e.g., image too small)
            enhanced_patches.append(p.copy())

    # Segmentation Pipeline: Patch -> Lines -> Words -> Characters
    for patch in enhanced_patches:
        gray = to_grayscale(patch)
        lines = segment_lines(gray)
        
        for y1, y2 in lines:
            line_img = patch[y1:y2, :]
            words = segment_words_from_line(line_img)
            
            if not words: words = [line_img] # Fallback if no words found
            
            for w in words:
                chars = segment_chars_from_word(w, args.min_char_width)
                
                # If segmentation fails to find chars, treat the whole word as a char
                if not chars:
                    ch = resize_and_normalize_char(w, args.char_h, args.char_w)
                    X.append(ch)
                    y.append(label_to_index[label_from_filename(fp)])
                else:
                    for c in chars:
                        ch = resize_and_normalize_char(c, args.char_h, args.char_w)
                        X.append(ch)
                        y.append(label_to_index[label_from_filename(fp)])

# Convert lists to Numpy arrays for Keras
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Shuffle the entire dataset
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# One-hot encode the labels
y_cat = to_categorical(y, num_classes=len(labels))

# Split into Train and Validation sets
val_count = max(1, int(args.val_fraction * len(X)))
X_val, y_val = X[:val_count], y_cat[:val_count]
X_train, y_train = X[val_count:], y_cat[val_count:]

# Compute class weights to handle imbalanced datasets
# This ensures rare characters are penalized more heavily if missed
counts = Counter(y.tolist())
class_weight = {i: (len(y) / (len(counts) * counts[i])) for i in counts}

# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------


def build_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    
    # In-network augmentation (lighter than the pre-processing one)
    # This prevents the model from memorizing exact pixel locations
    x = layers.RandomRotation(0.05)(inp)
    x = layers.RandomTranslation(0.03, 0.03)(x)
    
    # Feature Extraction (Convolutional Blocks)
    for filters in [32, 64, 128, 256, 256]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x) # Stabilizes training
        x = layers.MaxPool2D()(x)          # Reduces spatial dimensions
        x = layers.Dropout(0.25)(x)        # Randomly disable neurons to prevent overfitting
        
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x) # Flattens 3D volume to 1D vector
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)             # Heavy dropout before final layer
    out = layers.Dense(num_classes, activation="softmax")(x) # Softmax for probability distribution
    
    return models.Model(inp, out)

model = build_model((args.char_h, args.char_w, 3), len(labels))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
# Checkpoint: Save only the best model (based on validation accuracy)
ckpt = callbacks.ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True, verbose=1)

# EarlyStopping: Stop training if validation accuracy doesn't improve for 12 epochs
early = callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1)

# Reduce LR: If validation loss plateaus, reduce the learning rate to fine-tune
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    shuffle=True,
    class_weight=class_weight, # Apply the computed weights
    callbacks=[ckpt, rlr, early]
)

# ---------------------------------------------------------
# Saving Artifacts
# ---------------------------------------------------------
model.save(args.model_out)
with open(args.labels_out, "w") as f:
    json.dump(labels, f, indent=2)
print(f"Saved model to {args.model_out} and labels to {args.labels_out}")