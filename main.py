import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# IMPORTAR FUNCIONES COMUNES
# ===============================

from common import resize, medir_tiempo, mostrar_video_detectado

# ===============================
# IMPORTAR MÉTODOS ESPECÍFICOS
# ===============================

import metodo1     # líneas largas
import metodo2     # color
import metodo3     # clustering
import metodo4     # suavizado temporal
import metodo5     # lo que tengas en el script5


# ===============================
# SELECCIÓN DEL MÉTODO
# ===============================

METODOS = {
    "lineas_largas": metodo1.procesado_cpu,
    "color": metodo2.procesado_cpu,
    "clustering": metodo3.procesado_cpu,
    "suavizado": metodo4.procesado_cpu,
    "mixto": metodo5.procesado_cpu
}

# ===============================
# MAIN
# ===============================

def main():
    video_path = "./video_0.mp4"

    print("Métodos disponibles:")
    for k in METODOS:
        print(f" - {k}")

    metodo = input("\nElige método: ").strip()
    if metodo not in METODOS:
        print("Método no válido")
        return

    funcion = METODOS[metodo]

    print("\nMidiendo rendimiento...")
    t_cpu = medir_tiempo(video_path, funcion)
    fps_cpu = 1 / t_cpu
    print(f"CPU: {t_cpu:.4f}s por frame (~{fps_cpu:.1f} FPS)")

    # Gráfico de resultados
    data = {"Modo": ["CPU"], "FPS": [fps_cpu]}
    sns.barplot(data=data, x="Modo", y="FPS", hue="Modo",
                palette="viridis", legend=False)
    plt.title(f"Rendimiento método '{metodo}'")
    plt.ylabel("FPS")
    plt.show()

    # Mostrar video procesado
    print("\nReproduciendo video con líneas detectadas...")
    mostrar_video_detectado(video_path, funcion)


if __name__ == "__main__":
    main()
