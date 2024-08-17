from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')
img = cv2.imread('image/apple.jpg')
results = model(img, classes=47)  #  47: 'apple' 「print(model.names)で出力した値を参照」

class_list = results[0].boxes.cls.tolist()  # [47.0, 47.0]
count = len(class_list)

print(f'appleの数: {count}')
