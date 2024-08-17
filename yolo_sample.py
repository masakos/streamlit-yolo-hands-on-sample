from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')
print(model.names)  # YOLOが検出できるclass names のリスト

img = cv2.imread('image/apple.jpg')
results = model(img)  # class 'ultralytics.engine.results.Results のリストが返される

# https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results
annotated_image = results[0].plot()

cv2.imshow('annotated image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()  # qで終了
