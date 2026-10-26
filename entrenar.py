from ultralytics import YOLO

def main():
    # 1. Cargamos el best.pt presente en la raíz del proyecto
    model = YOLO("best.pt", task="pose")

    results = model.train(
        data="data.yaml",
        epochs=40,             # 40 épocas completas para asimilar perfiles y nuevos fondos
        imgsz=640,
        batch=4,
        device="cpu",          # Cambia a 0 si utilizas tarjeta NVIDIA
        workers=0,             # Evita congelamientos y deadlocks de subprocesos en Windows
        freeze=0,              # Descongelado para que aprenda contornos y siluetas laterales
        optimizer="AdamW",     # Mayor estabilidad y precisión en fine-tuning
        lr0=0.0003,            # Tasa moderada para evitar descalibrar pesos previos
        lrf=0.01,
        patience=0,            # Desactiva Early Stopping: garantiza las 40 épocas completas
        # Balance de funciones de coste
        cls=1.8,               # Castiga severamente la confusión entre clases (0 vs 1)
        box=8.5,
        pose=6.5,              # Reduce la exigencia en keypoints tapados por capuchas
        # Aumentos de datos
        fliplr=0.5,            # Invierte horizontalmente para duplicar perfiles izquierda/derecha
        hsv_h=0.04,
        hsv_s=0.7,
        hsv_v=0.5,             # Variación de iluminación para evitar sesgos por ropa oscura
        mosaic=0.5,            # Variedad de escalas y fondos
        scale=0.3,
        name="modelo_pose_perfiles_v3",
        plots=True
    )

if __name__ == '__main__':
    main()