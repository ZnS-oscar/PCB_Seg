from ultralytics import YOLO
import os
os.getcwd()
# Load the YOLO model
model = YOLO("demo/models/synpcbseg.pt")

# Export the model to ONNX format
model.export(format="onnx",dynamic=True)