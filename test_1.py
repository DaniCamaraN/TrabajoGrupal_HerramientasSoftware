import math
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

    return cv2.bitwise_and(img, mask)



def filtrar_lineas(lines):
    filtradas = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # avoid division by zero
            pendiente = (y2 - y1) / (x2 - x1)
            angulo = np.degrees(np.arctan(pendiente))

            # Keep diagonal lines only
            if 20 < abs(angulo) < 70:
                filtradas.append([x1, y1, x2, y2])

    return filtradas




def procesado_cpu(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)


    edges = cv2.Canny(blur, 20, 60)


    roi = region_interes2(edges)


    lines = cv2.HoughLinesP(
        roi, 1, np.pi / 180, 50,
        minLineLength=50, maxLineGap=100
    )


    filtradas = filtrar_lineas(lines)


    filtradas = sorted(
        filtradas,
        key=lambda x: math.dist((x[0], x[1]), (x[2], x[3])),
        reverse=True
    )[:10]


    output = frame.copy()

    for x1, y1, x2, y2 in filtradas:
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return output




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




def mostrar_video_detectado(video_path, funcion):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error al abrir el video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        frame_procesado = funcion(frame)

        # Optional: resize window for visibility
        frame_procesado = cv2.resize(frame_procesado, (960, 540))

        cv2.imshow("Detección de líneas (resultado final)", frame_procesado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    video_path = "Dataset/Video_Night2.mp4"

    print("Midiendo rendimiento...")
    t_cpu = medir_tiempo(video_path, procesado_cpu)
    fps_cpu = 1 / t_cpu

    print(f"CPU: {t_cpu:.4f}s por frame (~{fps_cpu:.1f} FPS)")

    # Graph
    data = {
        "Modo": ["CPU"],
        "FPS": [fps_cpu]
    }

    sns.barplot(data=data, x="Modo", y="FPS", hue="Modo", palette="viridis", legend=False)
    plt.title("Comparación de rendimiento CPU")
    plt.ylabel("Frames por segundo")
    plt.show()

    print("Reproduciendo video con líneas detectadas... (presiona 'q' para salir)")
    mostrar_video_detectado(video_path, procesado_cpu)
