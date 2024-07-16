import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.svm import LinearSVC
from typing import List, Dict
from dataset_reader import get_seizure_array
from FVfunctions import time2seg, get_interval_range, FIRfilterBP, IIRfilterBS, pot4signals, BANDAS, getMeFeatures
from features_stats import stats_features
import pandas as pd


def online_signal_test(model: LinearSVC,
                       channels: List[np.ndarray],
                       ch_names: List[str],
                       ex_seiz: Dict[str, str],
                       name: str,
                       fs: int = 512,
                       t_ant: int = 300,
                       t_pos: int = 60) -> None :
    """
    Plotea pintando de rojo los eventos epilépticos en un canal
    :param t_pos: tiempo posterior a la seizure para ver
    :param t_ant: tiempo antes de la seizure a analizar
    :param model: modelo pre-entrenado
    :param channels: señales leidas de diversos canales de EEG
    :param ch_names: nombres de los canales involucrados
    :param ex_seiz: un evento epiléptico, en forma de diccionario
    :param name: nombre del canal
    :param fs: frecuencia de muestreo
    """
    mtx_t_reg = np.array([ex_seiz['registration_start_time'], ex_seiz['registration_end_time']])
    arr_mtx_t_epi = get_seizure_array([ex_seiz])[0]
    t_epi_start = np.uint32(time2seg(time=arr_mtx_t_epi[0], ref_time=mtx_t_reg[0]))
    t_epi_end = np.uint32(time2seg(time=arr_mtx_t_epi[1], ref_time=mtx_t_reg[0]))
    i_start = int(fs * (t_epi_start - t_ant))
    if i_start < 0:
        print('epilepsia demasiado temprana, se aborta')
        return
    i_end = int(fs * (t_epi_end + t_pos))
    channels = channels[:, i_start : i_end]
    ex_channel = channels[0]

    # Array de instantes
    start = 0
    stop = (1 / fs) * len(ex_channel)
    arr_t = np.arange(start=start, stop=stop, step=(1 / fs))

    # 1: Se determinan los índices iniciales de los segmentos
    indexes = np.uint32(get_interval_range(0, t_epi_end + t_pos - t_epi_start + t_ant, fs, winlen=2))
    step = int(fs * 2)

    # 3: Se obtienen los segmentos, tanto de la señal como del array de instantes
    print(f"Segmentando las señales de {name}...")
    arr_seg_sig = [np.array([sig[idx: idx+step] for idx in indexes]) for sig in channels]
    arr_seg_t = np.array([arr_t[idx:(idx + step)] for idx in indexes])

    # 5: Se determinan los fv de estos segmentos
    print(f"Extrayendo las features de {name}...")
    df_fv = getMeFeatures(arr_seg_sig, ch_names, fs)

    # 6: Se predicen las labels con el modelo
    predictions = np.uint8(model.predict(df_fv))

    # 7: A partir de las labels predecidas se arman 2 'line_collection'

    # 7.1: Líneas true
    print(f"Prediciendo en {name}...")
    disp_channel = arr_seg_sig[0]
    arr_lineas_true = [np.column_stack((arr_seg_t[i], disp_channel[i])) for i in range(len(predictions)) 
                       if predictions[i] == 1]
    line_collection_true = LineCollection(arr_lineas_true, colors="red", linewidths=0.5, label='Epilepsia detectada')

    # 7.2: Líneas false
    arr_lineas_false = [np.column_stack((arr_seg_t[i], disp_channel[i])) for i in range(len(predictions)) 
                        if predictions[i] == 0]
    line_collection_false = LineCollection(arr_lineas_false, colors="green", linewidths=0.5, label='Sin epilepsia')

    # 8: Ploteo de los segmentos
    print(f"Ploteando en {name}...")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.add_collection(line_collection_true)
    ax.add_collection(line_collection_false)

    plt.axvline(x=arr_t[int(t_ant*fs)], linestyle="-.", linewidth=1.25, color="black", label='Inicio del ataque')
    plt.axvline(x=arr_t[int(-t_pos*fs)], linestyle="-.", linewidth=1.25, color="black", label='Final del ataque')

    plt.title(f"Predicciones en {name.strip('.edf')}", fontsize=12)
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Magnitud [mV]")
    plt.legend(loc='upper left', fontsize=8, framealpha=1.)
    plt.grid(linestyle='--')
    ax.autoscale_view()
    plt.show()