import cv2
import numpy as np
import math

from common import region_interes2, filtrar_lineas
from common_gpu import gpu_grayscale, gpu_blur


def procesado_cpu(frame):
    # 1. Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Suavizado
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Detección de bordes
    edges = cv2.Canny(blur, 20, 60)

    # 4. Región de interés
    roi = region_interes2(edges)

    # 5. HoughLinesP
    lines = cv2.HoughLinesP(
        roi, 1, np.pi / 180, 50,
        minLineLength=50, maxLineGap=100
    )

    # 6. Filtrar solo diagonales
    filtradas = filtrar_lineas(lines)

    # 7. Ordenar por longitud
    filtradas = sorted(
        filtradas,
        key=lambda x: math.dist((x[0], x[1]), (x[2], x[3])),
        reverse=True
    )[:10]

    # 8. Dibujar sobre la imagen original
    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return output


def procesado_gpu(frame):
    # ----------------------------
    # 1. GPU: Escala de grises
    # ----------------------------
    gray = gpu_grayscale(frame)

    # ----------------------------
    # 2. GPU: Suavizado
    # ----------------------------
    blur = gpu_blur(gray)

    # ----------------------------
    # 3. CPU: Canny
    #    (OpenCV no tiene Canny GPU)
    # ----------------------------
    edges = cv2.Canny(blur, 20, 60)

    # ----------------------------
    # 4. Región de interés (CPU)
    # ----------------------------
    roi = region_interes2(edges)

    # ----------------------------
    # 5. HoughLinesP (CPU)
    # ----------------------------
    lines = cv2.HoughLinesP(
        roi, 1, np.pi / 180, 50,
        minLineLength=50, maxLineGap=100
    )

    # ----------------------------
    # 6. Filtrar diagonales
    # ----------------------------
    filtradas = filtrar_lineas(lines)

    # ----------------------------
    # 7. Ordenar por longitud (CPU)
    # ----------------------------
    filtradas = sorted(
        filtradas,
        key=lambda x: math.dist((x[0], x[1]), (x[2], x[3])),
        reverse=True
    )[:10]

    # ----------------------------
    # 8. Dibujar resultado
    # ----------------------------
    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return output

