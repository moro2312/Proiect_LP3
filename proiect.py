"""
Detectarea semnelor de circulație

Cod inspirat:

Generative - ChatGPT

OpenCV - https://pypi.org/project/opencv-python/

Codul YOLOv8: https://github.com/ultralytics/ultralytics

Documentația oficială Ultralytics: https://docs.ultralytics.com/


"""

import cv2
import os
import glob
from ultralytics import YOLO

# Creează folderul "results" dacă nu există
os.makedirs("results", exist_ok=True)

# Încarcă modelul YOLOv8
model = YOLO("yolov8n.pt")  # Poți înlocui cu un model antrenat pe semne de circulație
names = model.names  # Dicționarul cu numele claselor

# Încarcă toate imaginile din folderul "images"
image_paths = glob.glob("image1.png")

# Funcție pentru a adăuga fundal text (pentru lizibilitate)
def draw_text(img, text, pos, font_scale=0.6, font_thickness=2, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - text_h - 4), (x + text_w + 2, y + 4), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Iterează prin fiecare imagine
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Eroare la citirea imaginii: {img_path}")
        continue

    results = model(img)
    filename = os.path.basename(img_path)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label_name = names[int(cls)]
            label_text = f"{label_name} ({conf:.2f})"
            coord_text = f"[{x1}, {y1}], [{x2}, {y2}]"

            # ✅ Afișează în consolă
            print(f"[{filename}] {label_text} - Coordonate: {coord_text}")

            # ✅ Desenează bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ✅ Etichetă și coordonate stilizate pe imagine
            draw_text(img, label_text, (x1, y1 - 30), font_scale=0.7, bg_color=(0, 128, 0))
            draw_text(img, coord_text, (x1, y1 - 10), font_scale=0.6, bg_color=(50, 50, 50))

    # Salvează imaginea în "results"
    cv2.imwrite(f"results/{filename}", img)

print("✅ Detecția a fost finalizată cu succes.")
