import cv2
import numpy as np
import time

def resize(img):
    return cv2.resize(img, (1024, 600), interpolation=cv2.INTER_AREA)

def region_interes(img):
    alto = img.shape[0]
    ancho = img.shape[1]
    # Define un polígono en forma de trapecio (parte inferior del frame)
    vertices = np.array([[
        (0, alto),
        (ancho//2 - 100, alto//2 + 50),
        (ancho//2 + 100, alto//2 + 50),
        (ancho, alto)
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def region_interes2(img, margen_horizontal=0.05, margen_inferior=0.05):
    alto, ancho = img.shape[:2]
    x_min = int(ancho * margen_horizontal)
    x_max = int(ancho * (1 - margen_horizontal))
    y_min = alto // 2
    y_max = int(alto * (1 - margen_inferior))
    mask = np.zeros_like(img)
    mask[y_min:y_max, x_min:x_max] = 255
    return cv2.bitwise_and(img, mask)

def filtrar_lineas(lines):
    filtradas = []
    if lines is None:
        return filtradas
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if x2 == x1:
            continue
        pendiente = (y2-y1)/(x2-x1)
        angulo = np.degrees(np.arctan(pendiente))
        if 20 < abs(angulo) < 70:
            filtradas.append([x1,y1,x2,y2])
    return filtradas

def medir_tiempo(video_path, funcion, n_frames=60, return_all=False):
    cap = cv2.VideoCapture(video_path)
    tiempos = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize(frame)
        inicio = time.time()
        _ = funcion(frame)
        tiempos.append(time.time() - inicio)
    cap.release()
    if return_all:
        return tiempos
    return np.mean(tiempos)


def mostrar_video_detectado(video_path, funcion):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir video")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize(frame)
        frame_procesado = funcion(frame)
        cv2.imshow("Detección (CPU)", frame_procesado)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def mostrar_video_detectado_doble(video_path, func_cpu, func_gpu):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize(frame)

        # Procesar ambos frames
        out_cpu = func_cpu(frame)
        out_gpu = func_gpu(frame)

        # Añadir texto identificativo
        cv2.putText(out_cpu, "CPU", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(out_gpu, "GPU", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Concatenar horizontalmente
        combined = np.hstack((out_cpu, out_gpu))

        cv2.imshow("Detección CPU vs GPU", combined)

        # Esperar 1ms y salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

