import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

from deploy.utils import (
    preprocess_image,
    postprocess,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "ONNX_model.onnx"

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(
#     BASE_DIR,
#     "..",
#     "models",
#     "ONNX_model.onnx"
# )


class ONNXInference:

    def __init__(self):

        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image):

        input_tensor, original_size = preprocess_image(image)

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor
            }
        )

        predictions = postprocess(
            outputs[0],   # boxes
            outputs[1],   # logits
            outputs[2],   # masks
            original_size
        )

        return predictions


# -------------------------------------------------
# Singleton model instance
# -------------------------------------------------

model = ONNXInference()