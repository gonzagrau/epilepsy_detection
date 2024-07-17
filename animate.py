import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation


def animate_signal_reading(arr_seg_true, arr_seg_false, arr_all):

    frames = len(np.concatenate((arr_seg_true, arr_seg_false)))
    line_collection_true = LineCollection(arr_seg_false, linewidths=1, colors='green', label='Epilepsia detectada')
    line_collection_false = LineCollection(arr_seg_true, linewidths=1, colors='red', label='Sin epilepsia')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.add_collection(line_collection_true)
    ax.add_collection(line_collection_false)
    ax.autoscale_view()
    ax.set_title('Clasificación en tiempo real')
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Magnitud [mV]")
    plt.legend(loc='upper left', fontsize=8, framealpha=1.)

    # Initialization function: plot the background of each frame
    def init():
        line_collection_true.set_segments([])
        line_collection_false.set_segments([])
        return line_collection_true, line_collection_false

    # Animation function: this is called sequentially
    def animate(i):
        segments_true = []
        segments_false = []
        for j in range(i):
            for seg in arr_seg_false:
                found = False
                if np.all(arr_all[j] == seg):
                    segments_false.append(arr_all[j])
                    found = True
                    break
            if not found:
                segments_true.append(arr_all[j])

        line_collection_false.set_segments(segments_true)
        line_collection_true.set_segments(segments_false)
        return line_collection_true, line_collection_false

    # Call the animator
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)
    anim.save(r'imagenes/animacion_tiempo_real_leyenda.gif', writer='imagemagick')
