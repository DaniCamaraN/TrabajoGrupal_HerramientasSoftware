import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from common import resize, medir_tiempo, mostrar_video_detectado_doble

import metodo1
import metodo2
import metodo3
import metodo4
import metodo5

# Métodos CPU y GPU en paralelo
METODOS = {
    "lineas_largas": (metodo1.procesado_cpu, metodo1.procesado_gpu),
    "color":         (metodo2.procesado_cpu, metodo2.procesado_gpu),
    "clustering":    (metodo3.procesado_cpu, metodo3.procesado_gpu),
    "suavizado":     (metodo4.procesado_cpu, metodo4.procesado_gpu),
    "mixto":         (metodo5.procesado_cpu, metodo5.procesado_gpu),
}

def main():
    video_path = "./video_0.mp4"

    print("Métodos disponibles:")
    for k in METODOS:
        print(f" - {k}")

    metodo = input("\nElige método: ").strip()
    if metodo not in METODOS:
        print("Método no válido")
        return

    func_cpu, func_gpu = METODOS[metodo]

    print("\nMidiendo rendimiento...")

    # CPU
    t_cpu = medir_tiempo(video_path, func_cpu)
    fps_cpu = 1 / t_cpu
    print(f"CPU: {t_cpu:.4f}s por frame (~{fps_cpu:.1f} FPS)")

    # GPU
    t_gpu = medir_tiempo(video_path, func_gpu)
    fps_gpu = 1 / t_gpu
    print(f"GPU: {t_gpu:.4f}s por frame (~{fps_gpu:.1f} FPS)")

    # === Gráfico FPS CPU vs GPU ===
    data = {
        "Modo": ["CPU", "GPU"],
        "FPS": [fps_cpu, fps_gpu]
    }

    sns.barplot(data=data, x="Modo", y="FPS", hue="Modo",
                palette="viridis", legend=False)
    plt.title(f"Rendimiento método '{metodo}'")
    plt.ylabel("FPS")
    plt.show()

    print("\nReproduciendo video con selector CPU/GPU...")
    print("Pulsa 'c' para CPU, 'g' para GPU, 'q' para salir")
    mostrar_video_detectado_doble(video_path, func_cpu, func_gpu)

if __name__ == "__main__":
    main()
