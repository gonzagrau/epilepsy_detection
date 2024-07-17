import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation


def animate_signal_reading(arr_seg_true, arr_seg_false):

    line_collection_true = LineCollection(arr_seg_false, linewidths=1, colors='green')
    line_collection_false = LineCollection(arr_seg_true, linewidths=1, colors='red')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.add_collection(line_collection_true)
    ax.add_collection(line_collection_false)
    ax.autoscale_view()

    # Initialization function: plot the background of each frame
    def init():
        line_collection_true.set_segments([])
        line_collection_false.set_segments([])
        return line_collection_true, line_collection_false

    # Animation function: this is called sequentially
    def animate(i):
        line_collection_true.set_segments(arr_seg_false[:i])
        line_collection_false.set_segments(arr_seg_true[:i])
        return line_collection_true, line_collection_false

    # Call the animator
    anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=50, blit=True)
    anim.save('line_animation.gif', writer='imagemagick')

    # Display the animation
    plt.show()
