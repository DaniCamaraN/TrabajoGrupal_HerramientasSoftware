import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse

from common import medir_tiempo, mostrar_video_detectado_doble

import metodo1
import metodo2
import metodo3
import metodo4
import metodo5

#DataSet:  https://www.kaggle.com/datasets/sergiolvarezsilva/lane-detection-dataset-morelos-sergio

# Métodos CPU y GPU en paralelo
METODOS = {
    "lineas_largas": (metodo1.procesado_cpu, metodo1.procesado_gpu),
    "color":         (metodo2.procesado_cpu, metodo2.procesado_gpu),
    "clustering":    (metodo3.procesado_cpu, metodo3.procesado_gpu),
    "suavizado":     (metodo4.procesado_cpu, metodo4.procesado_gpu),
    "lineas_noche":         (metodo5.procesado_cpu, metodo5.procesado_gpu)
}

def graficar_simple(fps_cpu,fps_gpu,metodo):
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

def procesamiento_simple(video_path, func_cpu, func_gpu, metodo):
    # CPU
    t_cpu = medir_tiempo(video_path, func_cpu)
    fps_cpu = 1 / t_cpu
    print(f"CPU: {t_cpu:.4f}s por frame (~{fps_cpu:.1f} FPS)")

    # GPU
    t_gpu = medir_tiempo(video_path, func_gpu)
    fps_gpu = 1 / t_gpu
    print(f"GPU: {t_gpu:.4f}s por frame (~{fps_gpu:.1f} FPS)")

    graficar_simple(fps_cpu, fps_gpu, metodo)

def procesamiento_complejo(video_path, func_cpu, func_gpu, metodo):
    # -----------------------------
    # Medir tiempos y FPS por frame
    # -----------------------------
    tiempos_cpu = medir_tiempo(video_path, func_cpu, n_frames=60, return_all=True)
    tiempos_gpu = medir_tiempo(video_path, func_gpu, n_frames=60, return_all=True)

    fps_cpu = 1 / np.mean(tiempos_cpu)
    fps_gpu = 1 / np.mean(tiempos_gpu)

    print(f"CPU: {np.mean(tiempos_cpu):.4f}s/frame (~{fps_cpu:.1f} FPS)")
    print(f"GPU: {np.mean(tiempos_gpu):.4f}s/frame (~{fps_gpu:.1f} FPS)")

    # -----------------------------
    # Barplot FPS promedio
    # -----------------------------
    df_avg = pd.DataFrame({
        "Modo": ["CPU", "GPU"],
        "FPS": [fps_cpu, fps_gpu]
    })
    sns.barplot(x="Modo", y="FPS", data=df_avg,hue="Modo", palette="viridis", legend=False)
    plt.title(f"FPS promedio - Método '{metodo}'")
    plt.show()

    # -----------------------------
    # Lineplot evolución temporal
    # -----------------------------
    frames = list(range(len(tiempos_cpu)))
    sns.lineplot(x=frames, y=tiempos_cpu, label="CPU")
    sns.lineplot(x=frames, y=tiempos_gpu, label="GPU")
    plt.xlabel("Frame")
    plt.ylabel("Tiempo (s)")
    plt.title(f"Evolución del tiempo/frame - Método '{metodo}'")
    plt.legend()
    plt.show()

    # -----------------------------
    # Heatmap FPS por metodo y modo
    # -----------------------------
    fps_matrix = []
    for f_cpu, f_gpu in METODOS.values():
        fps_matrix.append([1 / medir_tiempo(video_path, f_cpu), 1 / medir_tiempo(video_path, f_gpu)])
    fps_matrix = np.array(fps_matrix).T  # filas: CPU/GPU, columnas: métodos

    sns.heatmap(fps_matrix, annot=True, fmt=".1f", cmap="viridis",
                yticklabels=["CPU", "GPU"], xticklabels=list(METODOS.keys()))
    plt.title("FPS por método y modo")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Comparación CPU-GPU en detección de líneas de tráfico.")
    parser.add_argument("--video_path", help="Ruta al video", default=r"./video_0.mp4")
    parser.add_argument("--procesamiento_simple",
                        help="Si se quiere un procesamiento simple de solo una grafica comparando CPU y GPU",
                        default=True, type=bool)
    parser.add_argument("--procesamiento_complejo",
                        help="Si se quiere un procesamiento más complejo con una gráfica de comparativa CPU y GPU de todos los metodos",
                        default=False, type=bool)
    args = parser.parse_args()

    video_path = args.video_path

    print("Métodos disponibles:")
    for k in METODOS:
        print(f" - {k}")

    metodo = input("\nElige método: ").strip()
    if metodo not in METODOS:
        print("Método no válido")
        return

    func_cpu, func_gpu = METODOS[metodo]

    print("\nMidiendo rendimiento...")

    if args.procesamiento_simple:
        print("Procesamiento simple...")
        procesamiento_simple(video_path, func_cpu, func_gpu, metodo)

    if args.procesamiento_complejo:
        print("Procesamiento complejo...")
        procesamiento_complejo(video_path, func_cpu, func_gpu, metodo)

    print("\nReproduciendo video CPU/GPU...")
    mostrar_video_detectado_doble(video_path, func_cpu, func_gpu)

if __name__ == "__main__":
    main()
