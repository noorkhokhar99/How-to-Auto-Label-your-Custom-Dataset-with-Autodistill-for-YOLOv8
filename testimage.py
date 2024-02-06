from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('last.pt')

# Run inference on 'bus.jpg' with arguments
model.predict(source=0, show=True)