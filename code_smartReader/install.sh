#!/bin/bash

# Install packages from requirements.txt
pip install --no-deps -r requirements.txt
mkdir -p ./assets/models && wget -O ./assets/models/embedder.tflite -q https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite
echo "Finished installing requirements"
