import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.svm import LinearSVC
from typing import List, Dict
from dataset_reader import get_seizure_array
from FVfunctions import time2seg, get_interval_range, FIRfilterBP, IIRfilterBS, pot4signals, BANDAS
from features_stats import stats_features
import pandas as pd


def online_test(model: LinearSVC,
                channel: np.ndarray,
                seizures: List[Dict[str, str]],
                name: str,
                fs: int = 512 ) -> None :
    """
    Plotea pintando de rojo los eventos epilépticos en un canal
    :param model: modelo pre-entrenado
    :param channel: señal leida de un canal de EEG
    :param seizures: lista o array con los eventos epilépticos conocidos
    :param name: nombre del canal
    :param fs: frecuencia de muestreo
    """
    global t_epi_start
    ex_seiz = seizures[0]
    mtx_t_reg = np.array([ex_seiz['registration_start_time'], ex_seiz['registration_end_time']])
    arr_mtx_t_epi = get_seizure_array(seizures)


    # Array de instantes
    start = 0
    stop = (1 / fs) * len(channel)
    arr_t = np.arange(start=start, stop=stop, step=(1 / fs))

    # 1: Se determinan los índices iniciales de los segmentos
    t_reg_start = time2seg(time=mtx_t_reg[0], ref_time=mtx_t_reg[0])
    t_reg_end = time2seg(time=mtx_t_reg[1], ref_time=mtx_t_reg[0])
    indexes = np.uint32(get_interval_range(t_reg_start, t_reg_end, fs, winlen=2))

    # 2: Se seleccion la señal que se quiere analizar
    sig = channel
    step = int(fs * 2)

    # 3: Se obtienen los segmentos, tanto de la señal como del array de instantes
    arr_seg_sig = np.array([sig[idx:(idx + step)] for idx in indexes])
    arr_seg_t = np.array([arr_t[idx:(idx + step)] for idx in indexes])

    # 4: Se filtran los segmentos de señal
    arr_filtered_seg_sig = FIRfilterBP(arr_seg_sig)
    arr_filtered_seg_sig = IIRfilterBS(arr_filtered_seg_sig)

    # 5: Se determinan los fv de estos segmentos

    # 5.1: Potencia
    more_pot = pot4signals(arr_filtered_seg_sig, fs, divisor=1)
    pot_names = [f"potAbs{band.capitalize()}" for band in BANDAS.keys()]

    # 5.2: Estadistica
    stat_data = stats_features(arr_filtered_seg_sig)
    stats_names = list(stat_data.values())[:-1]
    more_stat = stat_data["matriz de features stat"]

    # 5.3: Feature vector
    data_fv = np.hstack((more_pot, more_stat))
    columnas = pot_names + stats_names
    df_fv = pd.DataFrame(data_fv, columns=columnas)

    # 6: Se predicen las labels con el modelo
    predictions = np.uint8(model.predict(df_fv))
    print(predictions)

    # 7: A partir de las labels predecidas se arman 2 'line_collection'

    # 7.1: Líneas true
    arr_lineas_true = [np.column_stack((arr_seg_t[i], arr_seg_sig[i])) for i in range(len(predictions)) if
                       predictions[i] == 1]
    line_collection_true = LineCollection(arr_lineas_true, colors="red", linewidths=1)

    # 7.2: Líneas false
    arr_lineas_false = [np.column_stack((arr_seg_t[i], arr_seg_sig[i])) for i in range(len(predictions)) if
                        predictions[i] == 0]
    line_collection_false = LineCollection(arr_lineas_false, colors="green", linewidths=1)

    # 8: Ploteo de los segmentos
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