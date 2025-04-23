from ultralytics import YOLO
import cv2

# Încarcă modelul YOLOv5 (varianta mică)
model = YOLO('yolov5s.pt')

# Încarcă imaginea
image_path = 'imagine1.jpg'
img = cv2.imread(image_path)

# Rulează detecția
results = model(img)

# Creează o copie a imaginii pentru desenare
output_img = img.copy()

# Clasele care ne interesează (din COCO dataset)
clase_vehicule = ['car', 'truck', 'bus']

# Iterează prin detecții
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        if cls_name in clase_vehicule:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, cls_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Afișează imaginea rezultată
cv2.imshow('Vehicule detectate', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Salvează imaginea cu vehicule
cv2.imwrite('vehicule_detectate.jpg', output_img)