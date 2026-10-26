from collections import defaultdict, deque
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
historial_clases = defaultdict(lambda: deque(maxlen=15))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        device="cpu",
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.50,
        iou=0.45,
    )

    for r in results:
        # Dibujar ÚNICAMENTE los esqueletos (kpt=True) sin las cajas/etiquetas viejas de YOLO
        annotated_frame = r.plot(boxes=False, labels=False)

        if r.boxes is not None and r.boxes.id is not None:
            cajas = r.boxes.xyxy.cpu().numpy()
            clases = r.boxes.cls.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            kpt_confs = None
            if r.keypoints is not None and r.keypoints.conf is not None:
                kpt_confs = r.keypoints.conf.cpu().numpy()

            for idx, (box, cls_id, track_id, box_conf) in enumerate(
                zip(cajas, clases, track_ids, confs)
            ):
                # Regla facial anatómica
                if cls_id == 1 and kpt_confs is not None and len(kpt_confs) > idx:
                    conf_kpts = kpt_confs[idx]
                    orejas_visibles = conf_kpts[3] > 0.40 or conf_kpts[4] > 0.40
                    rostro_frontal = conf_kpts[0] > 0.50

                    if orejas_visibles or rostro_frontal:
                        cls_id = 0

                historial_clases[track_id].append(cls_id)
                votos = historial_clases[track_id]
                clase_estable = max(set(votos), key=votos.count)

                # Colores BGR: Azul para person, Rojo para encapuchado
                color = (0, 0, 255) if clase_estable == 1 else (255, 100, 0)
                nombre = model.names[clase_estable]

                x1, y1, x2, y2 = map(int, box)

                # 1. Dibujar el recuadro limpio
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # 2. Dibujar la etiqueta única corregida
                etiqueta = f"ID:{track_id} {nombre} {box_conf:.2f}"
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - 25),
                    (x1 + len(etiqueta) * 11, y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    etiqueta,
                    (x1 + 4, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    cv2.imshow("Tesis - Analisis de Comportamiento", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()