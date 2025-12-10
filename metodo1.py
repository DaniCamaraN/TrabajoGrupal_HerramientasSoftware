import cv2
import math
import numpy as np
from common import region_interes2, filtrar_lineas
from common_gpu import gpu_grayscale, gpu_blur

def procesado_cpu(frame):
    # 1. Escala de grises y suavizado
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Canny
    edges = cv2.Canny(blur, 50, 150)

    # 3. Región de interés
    roi = region_interes2(edges)

    # 4. Hough
    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180, 50,
        minLineLength=50, maxLineGap=150
    )

    # 5. Filtrado de líneas
    filtradas = filtrar_lineas(lines)

    # Ordenar por longitud, quedarse con las 10 más largas
    filtradas = sorted(
        filtradas,
        key=lambda x: math.dist((x[0], x[1]), (x[2], x[3])),
        reverse=True
    )[:10]

    # 6. Dibujar resultado
    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return output

def procesado_gpu(frame):
    # -----------------------
    # GPU: grayscale + blur
    # -----------------------
    gray = gpu_grayscale(frame)
    blur = gpu_blur(gray)

    # CPU: Canny (no existe Canny GPU en OpenCV)
    edges = cv2.Canny(blur, 50, 150)

    # ROI
    roi = region_interes2(edges)

    # Hough (CPU)
    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180, 50,
        minLineLength=50, maxLineGap=150
    )

    filtradas = filtrar_lineas(lines)

    # ordenar por longitud
    filtradas = sorted(
        filtradas,
        key=lambda x: math.dist((x[0], x[1]), (x[2], x[3])),
        reverse=True
    )[:10]

    # Dibujar
    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return output