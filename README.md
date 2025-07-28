# Smart Bin - Backend (FastAPI + PyTorch)

## Overview
This is the FastAPI backend for the Smart Bin project. It loads a trained ResNet-18 model to classify images into predefined trash categories.

## Features
- REST API endpoint /predict for image classification.
- Uses a fine-tuned ResNet-18 model.
- Handles image preprocessing and normalization.
- CORS middleware for frontend integration.

## Tech Stack
- FastAPI
- PyTorch
- Torchvision
- Pillow
- Uvicorn

## How It Works
1. Receives an uploaded image via /predict.
2. Converts bytes to a PIL image and preprocesses (resize, normalize).
3. Passes image to the model → predicts class probabilities.
4. Returns JSON with:\
       - label: predicted trash category\
       - confidence: prediction probability

### Endpoint Example
POST /predict\
Content-Type: multipart/form-data\
file=@example.png

### Response
{\
  "label": "plastic",\
  "confidence": 0.9473\
}
