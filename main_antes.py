import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def procesado(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar umbral para obtener imagen binaria
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = gray.copy()
    return img_contours, contours

def mostrar(img, contours):
    # Dibuja los contornos encontrados en la imagen
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # para dibujar en color
    cv2.drawContours(img_color, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Video", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = cv2.VideoCapture('./video_0.mp4')
    print("Video cargado")

    ret, frame = video.read()
    if not ret:
        print("Error al leer el video.")
        video.release()
        exit()

    # Filtrado gaussiano
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Procesado de imágenes
    img_contours, contours = procesado(frame)
    img_contours2, contours2 = procesado(blurred)

    # Mostrar resultados
    print("Mostrando contornos...")
    mostrar(img_contours, contours)
    mostrar(img_contours2, contours2)

    video.release()
