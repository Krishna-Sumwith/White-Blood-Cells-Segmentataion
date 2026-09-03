import cv2
import numpy as np

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

INPUT_HEIGHT = 432
INPUT_WIDTH = 432

TOPK = 300
CONF_THRESHOLD = 0.50

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ----------------------------------------------------
# Image Preprocessing
# ----------------------------------------------------

def preprocess_image(image):

    original_height, original_width = image.shape[:2]

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    image = image.astype(np.float32)

    image /= 255.0

    image = (image - MEAN) / STD

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(np.float32), (
        original_height,
        original_width
    )


# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cxcywh_to_xyxy(boxes):

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.stack(
        [x1, y1, x2, y2],
        axis=1
    )


# ----------------------------------------------------
# Mask -> Polygon
# ----------------------------------------------------

def mask_to_polygon(mask):

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return []

    contour = max(
        contours,
        key=cv2.contourArea
    )

    contour = contour.squeeze()

    if contour.ndim != 2:
        return []

    return contour.tolist()


# ----------------------------------------------------
# Postprocess
# ----------------------------------------------------

def postprocess(
    pred_boxes,
    pred_logits,
    pred_masks,
    original_size,
):

    H, W = original_size

    pred_boxes = pred_boxes[0]
    pred_logits = pred_logits[0]
    pred_masks = pred_masks[0]

    scores = sigmoid(pred_logits)

    num_queries, num_classes = scores.shape

    scores = scores.reshape(-1)

    k = min(TOPK, len(scores))

    topk_idx = np.argpartition(
        -scores,
        k - 1
    )[:k]

    topk_scores = scores[topk_idx]

    order = np.argsort(-topk_scores)

    topk_scores = topk_scores[order]
    topk_idx = topk_idx[order]

    query_idx = topk_idx // num_classes
    labels = topk_idx % num_classes

    boxes = pred_boxes[query_idx]

    boxes = cxcywh_to_xyxy(boxes)

    boxes[:, [0, 2]] *= W
    boxes[:, [1, 3]] *= H

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        W
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        H
    )

    selected_masks = pred_masks[query_idx]

    predictions = []

    for score, label, box, mask in zip(
        topk_scores,
        labels,
        boxes,
        selected_masks
    ):

        if score < CONF_THRESHOLD:
            continue

        mask = cv2.resize(
            mask.astype(np.float32),
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )

        mask = mask > 0

        polygon = mask_to_polygon(mask)

        predictions.append(
            {
                "bbox": box.astype(float).tolist(),
                "score": float(score),
                "class_id": int(label),
                "mask": polygon,
            }
        )

    return predictions