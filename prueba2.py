import math

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import seaborn as sns

#Combinación líneas largas con lineas de color blancas y amarillas

def mostrar(img):
    cv2.imshow("Detección de líneas (CPU)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

    # Cálculo de márgenes
    x_min = int(ancho * margen_horizontal)
    x_max = int(ancho * (1 - margen_horizontal))
    y_min = alto // 2
    y_max = int(alto * (1 - margen_inferior))

    # Crear máscara
    mask = np.zeros_like(img)
    mask[y_min:y_max, x_min:x_max] = 255

    # Aplicar máscara
    masked = cv2.bitwise_and(img, mask)
    return masked


def filtrar_lineas(lines):
    filtradas = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # evitar división por 0
            pendiente = (y2 - y1) / (x2 - x1)
            angulo = np.degrees(np.arctan(pendiente))
            # Mantener solo líneas con ángulo entre 20° y 70° (aprox diagonales)
            if 20 < abs(angulo) < 70:
                filtradas.append([x1, y1, x2, y2])
    return filtradas

# -------------------------------
# FUNCIONES DE PROCESAMIENTO
# -------------------------------

def procesado_cpu(frame):
    """Procesamiento en CPU: detección de líneas de carril usando segmentación por color + Canny + Hough."""

    # Convertir a HSV (más robusto a la iluminación)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Máscara para líneas blancas
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Máscara para líneas amarillas (varía según iluminación)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combinar ambas máscaras
    mask = cv2.bitwise_or(mask_white, mask_yellow)

    # Aplicar Canny sobre la máscara (no sobre toda la imagen)
    edges = cv2.Canny(mask, 50, 150)

    # Aplicar región de interés (ROI)
    roi = region_interes2(edges)

    # Detectar líneas por transformada de Hough
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=150
    )

    # Filtrar líneas por orientación (diagonales)
    filtradas = filtrar_lineas(lines)

    # Ordenar por longitud y quedarte con las más largas
    lines_sorted = sorted(
        filtradas,
        key=lambda x: math.dist((x[0], x[1]), (x[2], x[3])),
        reverse=True
    )
    filtradas_max = lines_sorted[:4]

    # Dibujar las líneas sobre el frame original
    output = frame.copy()
    for (x1, y1, x2, y2) in filtradas_max:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return output


# -------------------------------
# MEDICIÓN DE RENDIMIENTO
# -------------------------------

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

# -------------------------------
# MAIN
# -------------------------------

def mostrar_video_detectado(video_path, funcion):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar el frame
        frame_procesado = funcion(frame)

        # Mostrar frame procesado
        cv2.imshow("Detección de líneas (resultado final)", frame_procesado)

        # Espera 1 ms, y si se presiona 'q', salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./video_0.mp4"

    print("Midiendo rendimiento...")
    t_cpu = medir_tiempo(video_path, procesado_cpu)
    fps_cpu = 1 / t_cpu

    print(f"CPU: {t_cpu:.4f}s por frame (~{fps_cpu:.1f} FPS)")

    # -------------------------------
    # GRÁFICO DE RESULTADOS
    # -------------------------------
    data = {
        "Modo": ["CPU"],
        "FPS": [fps_cpu]
    }
    #sns.barplot(data=data, x="Modo", y="FPS", palette="viridis")
    sns.barplot(data=data, x="Modo", y="FPS", hue="Modo", palette="viridis", legend=False)
    plt.title("Comparación de rendimiento CPU vs GPU")
    plt.ylabel("Frames por segundo")
    plt.show()

    # -------------------------------
    # MOSTRAR DETECCIÓN DE LÍNEAS
    # -------------------------------

    print("Reproduciendo video con líneas detectadas... (presiona 'q' para salir)")
    mostrar_video_detectado(video_path, procesado_cpu)