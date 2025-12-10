import cv2
import numpy as np
from common import region_interes2, filtrar_lineas
from common_gpu import gpu_grayscale, gpu_blur

# Memoria para tracking
prev_izquierda = None
prev_derecha = None

def media_linea(a, b, alpha=0.7):
    if a is None:
        return b
    return (alpha * np.array(a) + (1 - alpha) * np.array(b)).astype(int)

def procesado_cpu(frame):
    global prev_izquierda, prev_derecha

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_interes2(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 40,
                            minLineLength=40, maxLineGap=200)

    filtradas = filtrar_lineas(lines)

    # Clasificar por pendiente
    izquierda = []
    derecha = []

    for x1, y1, x2, y2 in filtradas:
        pendiente = (y2 - y1) / (x2 - x1)
        if pendiente < 0:
            izquierda.append([x1, y1, x2, y2])
        else:
            derecha.append([x1, y1, x2, y2])

    # Seleccionar la más larga de cada lado
    left = max(izquierda, key=lambda x: np.hypot(x[2]-x[0], x[3]-x[1])) if izquierda else None
    right = max(derecha, key=lambda x: np.hypot(x[2]-x[0], x[3]-x[1])) if derecha else None

    # Suavizado temporal
    if left is not None:
        prev_izquierda = media_linea(prev_izquierda, left)
    if right is not None:
        prev_derecha = media_linea(prev_derecha, right)

    output = frame.copy()

    if prev_izquierda is not None:
        x1, y1, x2, y2 = prev_izquierda
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 4)

    if prev_derecha is not None:
        x1, y1, x2, y2 = prev_derecha
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 4)

    return output

def procesado_gpu(frame):
    global prev_izquierda, prev_derecha

    gray = gpu_grayscale(frame)
    blur = gpu_blur(gray)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_interes2(edges)
    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180, 40,
        minLineLength=40, maxLineGap=200
    )

    filtradas = filtrar_lineas(lines)
    izquierda = []
    derecha = []

    for x1, y1, x2, y2 in filtradas:
        pendiente = (y2 - y1) / (x2 - x1)
        (izquierda if pendiente < 0 else derecha).append([x1, y1, x2, y2])

    left = max(izquierda, key=lambda x: np.hypot(x[2]-x[0], x[3]-x[1])) if izquierda else None
    right = max(derecha, key=lambda x: np.hypot(x[2]-x[0], x[3]-x[1])) if derecha else None

    if left is not None:
        prev_izquierda = media_linea(prev_izquierda, left)
    if right is not None:
        prev_derecha = media_linea(prev_derecha, right)

    output = frame.copy()
    if prev_izquierda is not None:
        cv2.line(output, prev_izquierda[:2], prev_izquierda[2:], (0, 255, 0), 4)
    if prev_derecha is not None:
        cv2.line(output, prev_derecha[:2], prev_derecha[2:], (0, 255, 0), 4)

    return output