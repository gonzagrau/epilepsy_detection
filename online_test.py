import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.svm import LinearSVC
from typing import List, Dict
from dataset_reader import get_seizure_array
from FVfunctions import time2seg, get_interval_range, FIRfilterBP, IIRfilterBS, pot4signals, BANDAS, getMeFeatures
from features_stats import stats_features
import pandas as pd


def online_test(model: LinearSVC,
                channels: List[np.ndarray],
                ch_names: List[str],
                seizures: List[Dict[str, str]],
                name: str,
                fs: int = 512 ) -> None :
    """
    Plotea pintando de rojo los eventos epilépticos en un canal
    :param model: modelo pre-entrenado
    :param channels: señales leidas de diversos canales de EEG
    :param ch_names: nombres de los canales involucrados
    :param seizures: lista o array con los eventos epilépticos conocidos
    :param name: nombre del canal
    :param fs: frecuencia de muestreo
    """
    ex_channel = channels[0]
    ex_seiz = seizures[0]
    mtx_t_reg = np.array([ex_seiz['registration_start_time'], ex_seiz['registration_end_time']])
    arr_mtx_t_epi = get_seizure_array(seizures)


    # Array de instantes
    start = 0
    stop = (1 / fs) * len(ex_channel)
    arr_t = np.arange(start=start, stop=stop, step=(1 / fs))

    # 1: Se determinan los índices iniciales de los segmentos
    t_reg_start = time2seg(time=mtx_t_reg[0], ref_time=mtx_t_reg[0])
    t_reg_end = time2seg(time=mtx_t_reg[1], ref_time=mtx_t_reg[0])
    indexes = np.uint32(get_interval_range(t_reg_start, t_reg_end, fs, winlen=2))
    step = int(fs * 2)

    # 3: Se obtienen los segmentos, tanto de la señal como del array de instantes
    print(f"segmentando las señales de {name}...")
    arr_seg_sig = [np.array([sig[idx:(idx + step)] for idx in indexes]) for sig in channels]
    arr_seg_t = np.array([arr_t[idx:(idx + step)] for idx in indexes])
    print(arr_seg_sig[0].shape)
    # 5: Se determinan los fv de estos segmentos
    print(f"extrayendo las features de {name}...")
    df_fv = getMeFeatures(arr_seg_sig, ch_names, fs)

    # 6: Se predicen las labels con el modelo
    predictions = np.uint8(model.predict(df_fv))
    print(predictions)

    # 7: A partir de las labels predecidas se arman 2 'line_collection'

    # 7.1: Líneas true
    print(f"prediciendo en {name}...")
    disp_channel = arr_seg_sig[0]
    arr_lineas_true = [np.column_stack((arr_seg_t[i], disp_channel[i])) for i in range(len(predictions)) if
                       predictions[i] == 1]
    line_collection_true = LineCollection(arr_lineas_true, colors="red", linewidths=1)

    # 7.2: Líneas false
    arr_lineas_false = [np.column_stack((arr_seg_t[i], disp_channel[i])) for i in range(len(predictions)) if
                        predictions[i] == 0]
    line_collection_false = LineCollection(arr_lineas_false, colors="green", linewidths=1)

    # 8: Ploteo de los segmentos
    print(f"ploteando en {name}...")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.add_collection(line_collection_true)
    ax.add_collection(line_collection_false)

    for i in range(len(arr_mtx_t_epi)):
        # 1.0- Obtención de instantes característicos del ataque epiléptico
        mtx_inst = arr_mtx_t_epi[i]

        t_epi_start = time2seg(time=mtx_inst[0], ref_time=mtx_t_reg[0])
        t_epi_end = time2seg(time=mtx_inst[1], ref_time=mtx_t_reg[0])

        plt.axvline(x=t_epi_start, linestyle=":", color="black")
        plt.axvline(x=t_epi_end, linestyle=":", color="black")

    plt.xlim((t_epi_start - 200, t_epi_end + 60))
    plt.title(f"{name}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Magnitud [mV]")
    ax.autoscale_view()
    plt.show()