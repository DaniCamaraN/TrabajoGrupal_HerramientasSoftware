import cv2
import numpy as np
from common import region_interes2, filtrar_lineas
from common_gpu import gpu_blur

def procesado_cpu(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filtros de color
    blanco_min = np.array([0, 0, 180])
    blanco_max = np.array([180, 30, 255])
    amarillo_min = np.array([15, 50, 100])
    amarillo_max = np.array([35, 255, 255])

    mask_white = cv2.inRange(hsv, blanco_min, blanco_max)
    mask_yellow = cv2.inRange(hsv, amarillo_min, amarillo_max)
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    # ROI
    roi = region_interes2(mask)

    # Canny
    edges = cv2.Canny(roi, 40, 120)

    # Hough
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 35,
        minLineLength=35, maxLineGap=200
    )

    filtradas = filtrar_lineas(lines)

    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 150, 255), 3)

    return output

def procesado_gpu(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blanco_min = np.array([0, 0, 180])
    blanco_max = np.array([180, 30, 255])
    amarillo_min = np.array([15, 50, 100])
    amarillo_max = np.array([35, 255, 255])

    mask_white = cv2.inRange(hsv, blanco_min, blanco_max)
    mask_yellow = cv2.inRange(hsv, amarillo_min, amarillo_max)
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    roi = region_interes2(mask)

    blur = gpu_blur(roi)
    edges = cv2.Canny(blur, 40, 120)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 35,
        minLineLength=35, maxLineGap=200
    )

    filtradas = filtrar_lineas(lines)

    output = frame.copy()
    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 150, 255), 3)

    return output