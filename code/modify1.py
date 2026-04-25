import cv2
import os
import numpy as np
from ultralytics import YOLO

# ==========================
# LOAD MODELS
# ==========================
if not os.path.exists("yolov8n.pt"):
    raise FileNotFoundError("[ERROR] YOLO weights 'yolov8n.pt' not found!")
if not os.path.exists("MobileNetSSD_deploy.prototxt"):
    raise FileNotFoundError("[ERROR] MobileNetSSD_deploy.prototxt not found!")
if not os.path.exists("MobileNetSSD_deploy.caffemodel"):
    raise FileNotFoundError("[ERROR] MobileNet weights 'MobileNetSSD_deploy.caffemodel' not found!")

yolo = YOLO("yolov8n.pt")
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


dynamic_classes = {0, 1, 2, 3, 5, 7}  # Set for O(1) lookup

CONF_THRESHOLD = 0.6

SHRINK_FACTOR = 0.8

# Dilation kernel — slightly expands the SHRUNK mask to cover object edges
# that the shrink may have missed. Net effect: core shrunk, edges softly covered.
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Boosts speed 3-5x on CPU with minimal accuracy loss for video sequences
FRAME_SKIP   = 5
frame_count  = 0
cached_boxes = []
# KITTI: baseline=0.54m, fx=718.856 → near objects ~60px, far objects ~10px
DISPARITY_MIN = 10   # pixels — far objects
DISPARITY_MAX = 60   # pixels — near objects

# ==========================
# FOLDERS
# ==========================
input_left_folder   = "/home/vm/00/image_0"   # LEFT  camera input
input_right_folder  = "/home/vm/00/image_1"   # RIGHT camera input
masked_left_folder  = "/home/vm/case/image_0"    # LEFT  masked output
masked_right_folder = "/home/vm/case/image_1"    # RIGHT masked output
mask_out_folder     = "dynamic_masks"     # binary masks saved separately
detect_folder       = "detections"
os.makedirs(masked_left_folder,  exist_ok=True)
os.makedirs(masked_right_folder, exist_ok=True)
os.makedirs(mask_out_folder,     exist_ok=True)
os.makedirs(detect_folder,       exist_ok=True)

print("[INFO] Starting combined YOLO + MobileNet pipeline...")
print(f"[INFO] Dynamic classes: {dynamic_classes}")
print(f"[INFO] Confidence threshold: {CONF_THRESHOLD}")
print(f"[INFO] Bounding box shrink factor: {SHRINK_FACTOR}")

# ==========================
# HELPER: Shrink a bounding box toward its center
# ==========================
def shrink_box(x1, y1, x2, y2, factor, img_w, img_h):
    """
    Shrinks a bounding box toward its center by `factor`.
    factor=1.0 → no change, factor=0.8 → 80% of original size.
    Returns clamped integer coordinates.
    """
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = (x2 - x1) / 2.0 * factor
    half_h = (y2 - y1) / 2.0 * factor
    sx1 = max(0, min(img_w, int(cx - half_w)))
    sy1 = max(0, min(img_h, int(cy - half_h)))
    sx2 = max(0, min(img_w, int(cx + half_w)))
    sy2 = max(0, min(img_h, int(cy + half_h)))
    return sx1, sy1, sx2, sy2

# ==========================
# PROCESS IMAGES
# ==========================
for img_name in sorted(os.listdir(input_left_folder)):

    # Skip non-image files
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    # Load LEFT image
    path = os.path.join(input_left_folder, img_name)
    image = cv2.imread(path)
    if image is None or image.size == 0:
        print(f"[WARNING] Skipping corrupt/empty left file: {img_name}")
        continue

    # Load RIGHT image
    right_path = os.path.join(input_right_folder, img_name)
    right_image = cv2.imread(right_path)
    if right_image is None or right_image.size == 0:
        print(f"[WARNING] Skipping corrupt/empty right file: {img_name}")
        continue

    original = image.copy()
    (h, w) = original.shape[:2]

    # ==================================================
    # 1. YOLO → DETECT DYNAMIC OBJECTS → BUILD MASK
    # ==================================================

    if frame_count % FRAME_SKIP == 0:
        results      = yolo(image, conf=CONF_THRESHOLD)[0]
        cached_boxes = results.boxes
    frame_count += 1

    # Binary mask: 255=static (SLAM uses these), 0=dynamic (SLAM ignores these)
    # This mask is saved separately so the patched ORB-SLAM2 extractor can load it.
    feature_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Inpaint masks (used to fill masked pixels so SLAM still gets some texture)
    inpaint_mask_left  = np.zeros((h, w), dtype=np.uint8)
    inpaint_mask_right = np.zeros((h, w), dtype=np.uint8)

    detected_dynamic = False

    for box in cached_boxes:
        cls = int(box.cls[0])

        if cls not in dynamic_classes:
            continue

        # Raw YOLO box — clamped to image bounds
        rx1 = max(0, min(w, int(box.xyxy[0][0])))
        ry1 = max(0, min(h, int(box.xyxy[0][1])))
        rx2 = max(0, min(w, int(box.xyxy[0][2])))
        ry2 = max(0, min(h, int(box.xyxy[0][3])))

        # Skip zero-area boxes
        if rx2 <= rx1 or ry2 <= ry1:
            continue

        # IMPROVEMENT: Shrink box to object core — preserves background features
        x1, y1, x2, y2 = shrink_box(rx1, ry1, rx2, ry2, SHRINK_FACTOR, w, h)

        # Mark same region in left inpaint mask
        inpaint_mask_left[y1:y2, x1:x2] = 255

        # Objects appear shifted LEFT in right image by their disparity amount
        drx1 = max(0, x1 - DISPARITY_MAX)
        drx2 = max(0, x2 - DISPARITY_MIN)
        drx1 = min(w, drx1)
        drx2 = min(w, drx2)
        inpaint_mask_right[y1:y2, drx1:drx2] = 255

        detected_dynamic = True

    # Apply small dilation to cover object edges after shrinking
    if detected_dynamic:
        inpaint_mask_left  = cv2.dilate(inpaint_mask_left,  DILATE_KERNEL, iterations=1)
        inpaint_mask_right = cv2.dilate(inpaint_mask_right, DILATE_KERNEL, iterations=1)

        # FIX 4: Correct feature_mask update — zero only the dynamic region
        # Previous erode() wrongly shrank the entire valid static area
        dynamic_region = cv2.dilate(inpaint_mask_left, DILATE_KERNEL, iterations=1)
        feature_mask[dynamic_region == 255] = 0

    if inpaint_mask_left.any():
        image = cv2.inpaint(image, inpaint_mask_left, 3, cv2.INPAINT_TELEA)
    if inpaint_mask_right.any():
        right_image = cv2.inpaint(right_image, inpaint_mask_right, 3, cv2.INPAINT_TELEA)

    # Save processed images for SLAM (inpainted — no black voids)
    cv2.imwrite(os.path.join(masked_left_folder,  img_name), image)
    cv2.imwrite(os.path.join(masked_right_folder, img_name), right_image)

    # Save binary feature mask separately (for Option 1 ORB-SLAM2 patch)
    cv2.imwrite(os.path.join(mask_out_folder, img_name), feature_mask)

    # ==================================================
    # 2. MOBILENET SSD → OBJECT DETECTION ON ORIGINAL IMAGE
    # ==================================================
  
    detect_img = original.copy()
    resized = cv2.resize(detect_img, (300, 300))
    blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    try:
        detections = net.forward()
    except cv2.error as e:
        print(f"[ERROR] MobileNet failed on {img_name}: {e}")
        continue

    if detections is None:
        print(f"[WARNING] No detections returned for {img_name}, skipping.")
        continue

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESHOLD:
            idx = int(detections[0, 0, i, 1])

            if idx >= len(CLASSES):
                print(f"[WARNING] Unknown class index {idx} in {img_name}, skipping.")
                continue

            label = CLASSES[idx]

            coords = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1 = max(0, int(coords[0]))
            y1 = max(0, int(coords[1]))
            x2 = min(w, int(coords[2]))
            y2 = min(h, int(coords[3]))

            # Skip zero-area boxes from clamping
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(detect_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

         
            text_y = max(15, y1 - 5)
            cv2.putText(detect_img, label, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(detect_folder, img_name), detect_img)

print("[INFO] Pipeline completed successfully.")
print(f"[INFO] Outputs:")
print(f"  Masked images (left) : {masked_left_folder}/")
print(f"  Masked images (right): {masked_right_folder}/")
print(f"  Binary masks         : {mask_out_folder}/")
print(f"  Detection visuals    : {detect_folder}/")
