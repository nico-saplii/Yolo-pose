from pathlib import Path

# Ajusta las rutas relativas exactas hacia donde están tus labels de entrenamiento y validación
carpetas = [
    Path("train/labels"),
    Path("valid/labels")  # o Path("val/labels") según tu estructura
]

total_modificados = 0

for carpeta in carpetas:
    if not carpeta.exists():
        print(f"Aviso: La carpeta {carpeta} no existe, revisa la ruta.")
        continue

    # 1. Borrar cualquier archivo de caché que esté dentro
    for cache_file in carpeta.glob("*.cache"):
        cache_file.unlink()
        print(f"Caché eliminada: {cache_file}")

    # 2. Modificar solo los archivos de personas encapuchadas (los que empiezan con 'enca')
    for archivo in carpeta.glob("enca*.txt"):
        with open(archivo, "r", encoding="utf-8") as f:
            lineas = f.readlines()

        nuevas_lineas = []
        for linea in lineas:
            partes = linea.strip().split()
            if partes and partes[0] == "0":
                partes[0] = "1"
                nuevas_lineas.append(" ".join(partes) + "\n")
            else:
                nuevas_lineas.append(linea)

        with open(archivo, "w", encoding="utf-8") as f:
            f.writelines(nuevas_lineas)
        
        total_modificados += 1

print(f"¡Proceso terminado! Se forzó la clase 1 en {total_modificados} archivos 'enca*.txt'.")