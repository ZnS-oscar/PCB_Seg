from ultralytics import YOLO
import cv2
# Load a model
# model = YOLO("/root/autodl-tmp/sahi/demo/models/synpcbseg.pt")  # Load the official YOLOv8n-seg model

model = YOLO("/root/autodl-tmp/jrspcbseg-release-1220/synpcbseg.pt")  # Load the official YOLOv8n-seg model



# Run inference on an image
results = model("/root/autodl-tmp/sahi/input/input/singlechip.png")  # Predict on an image

# Save the results image
annotated_frame = results[0].plot()

# 保存可视化结果
output_path = 'output_image.jpg'
cv2.imwrite(output_path, annotated_frame)