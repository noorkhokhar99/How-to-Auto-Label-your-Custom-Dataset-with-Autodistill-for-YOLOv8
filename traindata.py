from autodistill_yolov8 import YOLOv8


target_model = YOLOv8("yolov8n.pt")
target_model.train("./dataset/data.yaml", epochs=200)