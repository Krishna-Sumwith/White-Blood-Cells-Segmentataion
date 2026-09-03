from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import sys

# sys.path.insert(0, r"C:\Users\vishn\OneDrive\Documents\wb-cell-segmentation\deploy")

from deploy.inference import model

app = FastAPI(
    title="RF-DETR ONNX Segmentation API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "RF-DETR ONNX API is running."
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        image_bytes = await file.read()

        image_np = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image = cv2.imdecode(
            image_np,
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image."
            )

        predictions = model.predict(image)

        return {
            "num_detections": len(predictions),
            "predictions": predictions
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )