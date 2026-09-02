# White Blood Cell Instance Segmentation using RF-DETR
An end-to-end deep learning pipeline for White Blood Cell (WBC) detection, classification, and instance segmentation from peripheral blood smear images.

The project uses RF-DETR (RFDETRSegPreview) with a DINOv2 Vision Transformer backbone, trained using PyTorch and deployed through ONNX Runtime + FastAPI + Docker. TensorRT FP32 and FP16 engines were also generated to benchmark high-performance inference.

## Project Overview
Automated analysis of peripheral blood smear images can reduce manual effort and improve the consistency of white blood cell identification.

This project aims to:
- Detect individual white blood cells
- Classify cells into 10 different WBC classes
- Generate instance-level segmentation masks
- Export the trained model to ONNX
- Optimize the model using TensorRT
- Provide an inference API using FastAPI
- Containerize the deployment using Docker

