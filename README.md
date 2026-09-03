# White Blood Cell Instance Segmentation using RF-DETR
An end-to-end deep learning pipeline for White Blood Cell (WBC) detection, classification, and instance segmentation from peripheral blood smear images.

The project uses RF-DETR (RFDETRSegPreview) with a DINOv2 Vision Transformer backbone, trained using PyTorch and deployed through ONNX Runtime + FastAPI + Docker. TensorRT FP32 and FP16 engines were also generated to benchmark high-performance inference.

## Project Overview
Automated analysis of peripheral blood smear images can reduce manual effort, improve the consistency of white blood cell identification and holds significant potential for aiding doctors in disease diagnosis through blood tests

This project aims to:
- Detect individual white blood cells
- Classify cells into 10 different WBC classes
- Generate instance-level segmentation masks
- Export the trained model to ONNX
- Optimize the model using TensorRT
- Provide an inference API using FastAPI
- Containerize the deployment using Docker

## Tech Stack
- PyTorch
- RFDETRSegPreview
- ONNX
- TensorRT
- FastAPI
- Docker
- Python

## Dataset
**WBC Instance Segmentation & Classification**\
The project uses a Kaggle dataset containing peripheral blood smear images with instance segmentation annotations.
🔗[Download Dataset](https://www.kaggle.com/datasets/jimutbahanpal/wbc-instance-segmentation-and-classification)  

**WBC Classes**\
The dataset consists of ten distinct peripheral blood smear classes, each featuring multiple multi-class white blood cells per slide.
|Class         |                                        Description                                       |
|--------------|------------------------------------------------------------------------------------------|
|Blast Cell    |Immature cells that develop into blood cells, Absent in blood.                            |
|Promyelocyte  |Early stage in the formation of granulocytes, Absent in blood.                            |
|Myelocyte     |Intermediate stage in granulocyte development, Absent in blood.                           |
|Metamyelocyte |Late immature stage before band cells, Absent or very rare in blood.                      |
|Band Cell     |Nearly mature neutrophils that help fight infections, Small numbers in blood.             |
|Neutrophil    |Fight bacterial and fungal infections, 40-70% of WBC present in blood.                    |
|Lymphocyte    |Fight viral infections and produce antibodies, 20-40% of WBC present in blood.            |
|Monocyte      |Remove germs and dead cells, help immune response, 2-8% of WBC present in blood.          |
|Eosinophil    |Fight parasites and help in allergic reactions, 1-4% of WBC present in blood.             |
|Basophil      |Release histamine during allergic and inflammatory reactions, <1% of WBC present in blood.|

The data was split into
- 80% Training
- 20% Validation
 
Original training image counts varied significantly between classes.

Training Distribution Before Augmentation
|Class              | Train  |Validation|
|-------------------|--------|----------|
|Blast Cell         |   54   |    14    |
|Promyelocyte       |   29   |     7    |
|Myelocyte          |   66   |    17    |
|Metamyelocyte      |   25   |     6    |
|Band Cell          |   48   |    12    |
|Neutrophil         |   79   |    20    |
|Lymphocyte         |   48   |    12    |
|Monocyte           |   42   |    11    |
|Eosinophil         |   54   |    13    |
|Basophil           |   21   |     5    |

## Data Augmentation
Each class augmented so that it contained 80 training images.  
The following augmentations were applied:
- Horizontal Flip
- Rotation (-10° to +10°)
- Random Brightness & Contrast
- Gaussian Blur
- Random Gamma

These augmentations were used to improve robustness against variations in cell orientation, microscope illumination, image focus, staining, and camera conditions.

## Model
**RF-DETR**  
The selected architecture is:  
**RFDETRSegPreview (RF-DETR)** built on DINOv2 Vision Transformer backbone  

## Training Configuration
- Framework: Pytorch
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU Memory: 6 GB
- Epochs: 15
- Input Size: 432×432
- Batch Size: 1
- Gradient Accumulation: 8
- Learning Rate: 1e-4
- Mixed Precision: FP16
- Early Stopping: Disabled
- Normalization Mean = [0.485,0.456,0.406]
- Normalization Std = [0.229,0.224,0.225]

## Model Optimization
To improve inference performance
1. Pytorch model exported to ONNX
2. ONNX model converted to TensorRT engine
3. Quantization applied to reduce latency

RF-Deter internally supports ONNX-based conversation processing through its built-in conversation support command model.export().  
The ONNX model was executed using the CPUExecutionProvider.  
The exported ONNX model preserved the same evaluation metrics as Pytorch model.

## Model Conversion: ONNX to TensorRT
FP32 Engine
```
trtexec --onnx=ONNX_model.onnx --saveEngine=TensorRT_model.engine
```
FP16 Engine
```
trtexec --onnx=ONNX_model.onnx --saveEngine=TensorRT_model_fp16.engine --fp16
```

## Model Performance
The trained PyTorch model achieved the following validation metrics
|  Metric  |Score |
|----------|------|
|mAP@50    |0.8527|
|mAP@50:95 |0.8337|
|Mean IoU  |0.9550|
|Precision |0.8766|
|Recall    |0.9310|
|F1-Score  |0.9030|

### Model Benchmark
| Model       | Precision |   Size   |Images Tested| Total Time | Latency Time (ms)   |  FPS   |
|-------------|-----------|----------|-------------|------------|---------------------|--------|
| PyTorch-CPU | FP32      |132,506 KB|     117     |105.3401 sec| 900.34              |  1.11  |
| PyTorch-GPU | FP32      |132,506 KB|     117     |  8.6533 sec| 73.96               | 13.52  | 
| ONNX        | FP32      |133,201 KB|     117     | 86.8964 sec| 742.70              |  1.35  |
| TensorRT    | FP32      |125,415 KB|      20     |545.7800 ms | 27.29               | 36.65  |
| TensorRT    | FP16      | 64,518 KB|      20     |194.8700 ms | 9.74                | 102.62 |

## Sample Predictions
<p align="center">
  <img src="sample predictions/wbc_prediction-1.jpg" width="300">
  <img src="sample predictions/wbc_prediction-4.jpg" width="300">
</p>

<p align="center">
  <img src="sample predictions/wbc_prediction-6.jpg" width="300">
  <img src="sample predictions/wbc_prediction-7.jpg" width="300">
</p>

## Deployment
The ONNX model was deployed using
- ONNX Runtime
- FastAPI
- Swagger UI
- Docker

The deployment exposes the model through a REST API and was tested through Swagger UI before being containerized with Docker.

## Clone the Repository
```
git clone https://github.com/Krishna-Sumwith/White-Blood-Cells-Segmentataion.git
cd White-Blood-Cells-Segmentataion
```

## Running the API
Install dependencies to run without Docker
```
pip install -r requirements.txt
```
Start the FastAPI server:
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open Swagger UI:
```
http://127.0.0.1:8000/docs
```
Upload an image to receive the output in json format.

## Docker Deployment
Build Docker image:
```
docker build -t wbc .
```
Run container:
```
docker run -p 8000:8000 wbc
```
Open API Docs
```
http://127.0.0.1:8000/docs
```

## Example API Request
Endpoint:
POST /predict
Input:
- Image file

Output:
- The JSON file contains the number of detections, along with the bounding box, confidence score, class ID, and mask for each detection.
