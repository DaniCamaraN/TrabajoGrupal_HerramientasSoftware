import math
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# FUNCIONES AUXILIARES
# ===============================

def mostrar(img):
    cv2.imshow("Detección de líneas (CPU)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def region_interes2(img, margen_horizontal=0.05, margen_inferior=0.05):
    alto, ancho = img.shape[:2]
    x_min = int(ancho * margen_horizontal)
    x_max = int(ancho * (1 - margen_horizontal))
    y_min = alto // 2
    y_max = int(alto * (1 - margen_inferior))
    mask = np.zeros_like(img)
    mask[y_min:y_max, x_min:x_max] = 255
    masked = cv2.bitwise_and(img, mask)
    return masked

def filtrar_lineas(lines):
    filtradas = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            pendiente = (y2 - y1) / (x2 - x1)
            angulo = np.degrees(np.arctan(pendiente))
            if 20 < abs(angulo) < 70:
                filtradas.append([x1, y1, x2, y2])
    return filtradas

# ===============================
# SUAVIZADO TEMPORAL DE LÍNEAS
# ===============================

def suavizar_lineas(lineas_prev, lineas_nuevas, max_dist=50, alpha=0.3):
    """
    Suaviza la posición de las líneas detectadas entre frames consecutivos.
    Si una línea nueva está cerca de una anterior, se interpola entre ambas.
    """
    if lineas_prev is None:
        return lineas_nuevas

    lineas_suav = []
    usadas_prev = set()

    for ln in lineas_nuevas:
        x1n, y1n, x2n, y2n = ln
        cxn = (x1n + x2n) / 2
        cyn = (y1n + y2n) / 2

        # Buscar línea previa más cercana
        mejor_dist = float('inf')
        mejor_lp = None
        mejor_idx = -1
        for i, lp in enumerate(lineas_prev):
            if i in usadas_prev:
                continue
            x1p, y1p, x2p, y2p = lp
            cxp = (x1p + x2p) / 2
            cyp = (y1p + y2p) / 2
            dist = math.hypot(cxn - cxp, cyn - cyp)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_lp = lp
                mejor_idx = i

        # Si la línea es similar en posición, suavizar
        if mejor_lp is not None and mejor_dist < max_dist:
            usadas_prev.add(mejor_idx)
            x1p, y1p, x2p, y2p = mejor_lp
            x1s = int(alpha * x1p + (1 - alpha) * x1n)
            y1s = int(alpha * y1p + (1 - alpha) * y1n)
            x2s = int(alpha * x2p + (1 - alpha) * x2n)
            y2s = int(alpha * y2p + (1 - alpha) * y2n)
            lineas_suav.append([x1s, y1s, x2s, y2s])
        else:
            # Si no hay línea previa cercana, mantener la nueva
            lineas_suav.append(ln)

    return lineas_suav


# ===============================
# PROCESAMIENTO DE CADA FRAME
# ===============================

lineas_prev_global = None  # almacenará líneas del frame anterior

def procesado_cpu(frame):
    """Procesamiento en CPU: detección de líneas por Hough transform + suavizado temporal."""
    global lineas_prev_global

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI
    roi = region_interes2(edges)

    # Detección de líneas
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
    filtradas = filtrar_lineas(lines)

    # Ordenar por longitud y quedarse con las 10 más largas
    lines_sorted = sorted(
        filtradas,
        key=lambda x: math.dist((x[0], x[1]), (x[2], x[3])),
        reverse=True
    )
    filtradas_max = lines_sorted[:10]

    # 🔹 Aplicar suavizado temporal con líneas previas
    filtradas_suav = suavizar_lineas(lineas_prev_global, filtradas_max)
    lineas_prev_global = filtradas_suav  # actualizar para siguiente frame

    # Dibujar
    output = frame.copy()
    for (x1, y1, x2, y2) in filtradas_suav:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return output


# ===============================
# MEDICIÓN DE RENDIMIENTO
# ===============================

def medir_tiempo(video_path, funcion, n_frames=60):
    cap = cv2.VideoCapture(video_path)
    tiempos = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        inicio = time.time()
        _ = funcion(frame)
        fin = time.time()
        tiempos.append(fin - inicio)
    cap.release()
    return np.mean(tiempos)

# ===============================
# REPRODUCCIÓN DEL VIDEO
# ===============================

def mostrar_video_detectado(video_path, funcion):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_procesado = funcion(frame)
        cv2.imshow("Detección de líneas (resultado final)", frame_procesado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    video_path = "./video_0.mp4"

    print("Midiendo rendimiento...")
    t_cpu = medir_tiempo(video_path, procesado_cpu)
    fps_cpu = 1 / t_cpu

    print(f"CPU: {t_cpu:.4f}s por frame (~{fps_cpu:.1f} FPS)")

    data = {"Modo": ["CPU"], "FPS": [fps_cpu]}
    sns.barplot(data=data, x="Modo", y="FPS", hue="Modo", palette="viridis", legend=False)
    plt.title("Comparación de rendimiento CPU vs GPU")
    plt.ylabel("Frames por segundo")
    plt.show()

    print("Reproduciendo video con líneas detectadas... (presiona 'q' para salir)")
    mostrar_video_detectado(video_path, procesado_cpu)
