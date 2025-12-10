import cv2
import numpy as np
from common import region_interes2, filtrar_lineas
from common_gpu import gpu_grayscale, gpu_blur

def procesado_cpu(frame):
    # 1. Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Filtros para líneas blancas y amarillas
    blanco_min = np.array([0, 0, 180])
    blanco_max = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, blanco_min, blanco_max)

    amarillo_min = np.array([15, 50, 100])
    amarillo_max = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, amarillo_min, amarillo_max)

    mask = cv2.bitwise_or(mask_white, mask_yellow)

    # 3. ROI
    roi = region_interes2(mask)

    # 4. Canny
    edges = cv2.Canny(roi, 50, 150)

    # 5. Hough
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 30,
        minLineLength=30, maxLineGap=200
    )

    filtradas = filtrar_lineas(lines)

    # 6. Dibujar
    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 3)

    return output

def procesado_gpu(frame):
    # HSV solo CPU (transformación barata)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filtros de color CPU (rápidos)
    blanco_min = np.array([0, 0, 180])
    blanco_max = np.array([180, 30, 255])
    amarillo_min = np.array([15, 50, 100])
    amarillo_max = np.array([35, 255, 255])

    mask_white = cv2.inRange(hsv, blanco_min, blanco_max)
    mask_yellow = cv2.inRange(hsv, amarillo_min, amarillo_max)
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    roi = region_interes2(mask)

    # GPU: blur + grayscale (aunque mask ya es grises)
    blur = gpu_blur(roi)

    # Canny CPU
    edges = cv2.Canny(blur, 50, 150)

    # Hough CPU
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 30,
        minLineLength=30, maxLineGap=200
    )

    filtradas = filtrar_lineas(lines)

    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 3)

    return output