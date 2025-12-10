import cv2
import numpy as np
from sklearn.cluster import KMeans
from common import region_interes2, filtrar_lineas
from common_gpu import gpu_grayscale, gpu_blur

def procesado_cpu(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    roi = region_interes2(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 40,
                            minLineLength=30, maxLineGap=200)

    filtradas = filtrar_lineas(lines)

    if len(filtradas) < 2:
        return frame

    # Agrupar por pendiente (dos clusters: izquierda/derecha)
    pendientes = []
    for x1, y1, x2, y2 in filtradas:
        pendiente = (y2 - y1) / (x2 - x1)
        pendientes.append([pendiente])

    kmeans = KMeans(n_clusters=2, n_init='auto').fit(pendientes)

    grupo1 = []
    grupo2 = []

    for idx, line in enumerate(filtradas):
        if kmeans.labels_[idx] == 0:
            grupo1.append(line)
        else:
            grupo2.append(line)

    output = frame.copy()

    # Dibujar cada grupo con color distinto
    for x1, y1, x2, y2 in grupo1:
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for x1, y1, x2, y2 in grupo2:
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return output

def procesado_gpu(frame):
    gray = gpu_grayscale(frame)
    blur = gpu_blur(gray)

    edges = cv2.Canny(blur, 50, 150)

    roi = region_interes2(edges)
    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180, 40,
        minLineLength=30, maxLineGap=200
    )

    filtradas = filtrar_lineas(lines)
    if len(filtradas) < 2:
        return frame

    # clustering CPU (muy rápido, no vale la pena llevarlo a GPU)
    pendientes = [[(y2-y1)/(x2-x1)] for x1,y1,x2,y2 in filtradas]
    kmeans = KMeans(n_clusters=2, n_init='auto').fit(pendientes)

    grupo1, grupo2 = [], []
    for idx, line in enumerate(filtradas):
        if kmeans.labels_[idx] == 0:
            grupo1.append(line)
        else:
            grupo2.append(line)

    output = frame.copy()
    for x1, y1, x2, y2 in grupo1:
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for x1, y1, x2, y2 in grupo2:
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return output
