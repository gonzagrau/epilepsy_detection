import numpy as np
from mne import io
import random
from datetime import  datetime, timedelta
import scipy.signal as signal
import pandas as pd
from features_stats import stats_features
from typing import Tuple, List

# CONSTANTE: Bandas de ritmos cerebrales
BANDAS = {"delta": [0.5, 4],
          "theta": [4,8],
          "alpha": [8,13],
          "beta": [13,30],
          "gamma": [30,50]}


def time2seg(time: str, ref_time: str="00.00.00") -> np.uint32:
    """
    Función para calcular segundos a la hora 00.00.00
    """

    time_format = "%H.%M.%S"
    default_date = "2022-12-18"

    try:
        time_i = datetime.strptime(default_date + " " + ref_time, "%Y-%m-%d " + time_format)
        time_f = datetime.strptime(default_date + " " + time, "%Y-%m-%d " + time_format)
    except ValueError:
        alt_time_format = '%H.%M.%S'
        time_i = datetime.strptime(default_date + " " + ref_time, "%Y-%m-%d " + alt_time_format)
        time_f = datetime.strptime(default_date + " " + time, "%Y-%m-%d " + alt_time_format)

    # Si el tiempo final excede al inicial, hubo un salto de dia
    if time_f < time_i:
        time_f += timedelta(days=1)

    time_difference = time_f - time_i
    elapsed_seconds = time_difference.total_seconds()

    return np.uint32(elapsed_seconds)


def FIRfilterBP(arr_signals: np.ndarray,
                freq:List[int] | None = None,
                order: int=(2 ** 8) + 1,
                fs: int=512) -> np.ndarray:
    """
    Aplica un filtro pasabandas FIR vectorialmente a un array de señales
    :param arr_signals: matriz NxL, donde cada fila es una señal, cada columna una muestra temporal
    :param freq: lista de frecuencias de corte
    :param order: orden del filtro
    :param fs: frecuencia de muestreo
    :return: matrix NxL con las señales filtradas
    """
    if freq is None:
        freq = [1, 100]

    b_bp = signal.firwin(numtaps=order, cutoff=freq, window="blackman", pass_zero="bandpass", fs=fs)
    arr_filtered_signals = signal.filtfilt(b_bp, 1, arr_signals, axis=1)

    return arr_filtered_signals


def IIRfilterBS(arr_signals: np.ndarray,
                freq: List[int] | None=None,
                order: int=2,
                fs: int=512) -> np.ndarray:
    """
    Aplica un filtro pasabandas FIR vectorialmente a un array de señales
    (Ver docstring de FIRfilterBS() para mas informacion)
    """
    if freq is None:
        freq = [49.5, 50.5]

    b_bs, a_bs = signal.iirfilter(N=order, Wn=freq, btype="bandstop", ftype='butter', fs=fs, output='ba')
    arr_filtered_signals = signal.filtfilt(b_bs, a_bs, arr_signals, axis=1)

    return arr_filtered_signals


def closest(lst: List[float] | np.ndarray, K: float | int) -> float:
    """
    Finds the closest value in iterable (i.e: list, np.ndarray) 'lst' to the value 'K'
    """
    return lst[min(range(len(lst)), key = lambda i: abs(K-lst[i]))]


def pot4band(arr_freq: np.ndarray, psd: np.ndarray, tipo: str = 'absoluta') -> np.ndarray:
    """
    Calcula la potencia absoluta o relativa de cada banda frecuencial de ritmos cerebrales en una PSD
    :param arr_freq: bins de frecuencias en la PSD
    :param psd: power sprectral density estimada
    :param tipo: indica si es la potencia relativa o absoluta 
    :return: array 1D con la potencia de cada banda
    """
    # 0- Definición de lista de almacenamiento
    arr_pot_abs4band = []

    # 1- Cálculo de potencias absolutas y relativas para cada banda
    for banda in BANDAS.values():
        # 2- Encontrar los índices de las frecuencias dentro de la banda de interés
        cercano_min = closest(arr_freq, banda[0])
        cercano_max = closest(arr_freq, banda[1])

        i_fmin = np.uint32([i for i in range(len(arr_freq)) if arr_freq[i] == cercano_min][0])
        i_fmax = np.uint32([i for i in range(len(arr_freq)) if arr_freq[i] == cercano_max][0])

        # 3- Integrarla PSD dentro de la banda de frecuencia
        pot_abs_band = np.sum(psd[i_fmin:i_fmax]) * (arr_freq[1] - arr_freq[0])

        # 4- Almacenamiento del valor de potencia calculado
        arr_pot_abs4band.append(pot_abs_band)

    # 5- Preparación de arrays
    arr_pot_abs4band = np.array(arr_pot_abs4band)

    if tipo == 'relativa':
        arr_pot_abs4band = arr_pot_abs4band / np.sum(arr_pot_abs4band)
    
    return arr_pot_abs4band


def pot4signals(arr_signals: np.ndarray, fs: int=512, divisor: int=100,tipo: str = 'absoluta') -> np.ndarray:
    """
    Calcula la potencia en cada banda para un array de señales, estimando PSD con Welch
    :param arr_signals: matriz NxL donde cada fila es una señal de longitud L
    :param fs: frecuencia de muestro
    :param divisor: cantidad de segmentos para welch
    :param tipo: indica si es la potencia relativa o absoluta 
    :return:
    """
    # 0- Cálculo del nper
    nper = int(len(arr_signals[0]) // divisor)

    # 1- Cálculo del array de PSDs para cada señal del array de señales
    arr_freq = signal.welch(x=arr_signals[0], fs=fs, noverlap=nper // 2, nperseg=nper)[0]
    arr_psd = np.array([signal.welch(x=sig, fs=fs, noverlap=nper // 2, nperseg=nper)[1] for sig in arr_signals])

    # 2 Cálculo de potencias para cada señal del array de señales
    arr_pot4signals = np.array([pot4band(arr_freq, psd,tipo) for psd in arr_psd])

    return arr_pot4signals


def get_interval_range(t_start: int, t_end: int, fs: int, winlen: int) -> np.ndarray:
    """
    Crea un array que representa un slice temporal desde t_start hasta t_end con un ancho de winlen, los 3 en [s]
    :param t_start: tiempo inicial en segundos
    :param t_end: tiempo final en segundos
    :param fs: frecuencia de muestreo, en Hz
    :param winlen: longitud de la ventana temporal, en segundos
    :return: arange desde t_start hasta t_end expresado en MUESTRAS (i.e.: indices)
    """
    step = fs * winlen
    i_start = fs * t_start
    start = i_start
    stop = step * ((t_end - t_start) // winlen) + i_start
    interval = np.arange(start=start, stop=stop, step=step)
    return interval


def getMeData(sig: np.ndarray,
              mtx_t_reg: np.ndarray,
              arr_mtx_t_epi: np.ndarray,
              winlen: int=2,
              fs: int=512,
              proportion: float=0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segmenta la señal en intervalos de longitud winlen [s] clasificados como 1 o 0 (epilepsio o no epilepsia)
    :param sig: señal leida de un único canal EEG
    :param mtx_t_reg: array 2x3 con inicio y final de lectura en [hh, mm, ss]
    :param arr_mtx_t_epi: array Nx(2x3) con N inicios y finales de ataques epilépticos en [hh, mm, ss]
    :param winlen: longitud de los segmentos a extraer
    :param fs: frecuencia de muestreo
    :param proportion: proporcios de segmentos 'False' (la de 'True' sería 1.0 - proportion)
    :return: tupla con arr_seg (matriz de señales en cada fila) y labels (etiqueta de cada segmentos)
    """
    # 0- Declaración de parámetros importantes
    step = fs * winlen

    ###########################################################################
    # PARTE 1 - Obtención de índices iniciales de intervalos verdaderos
    ###########################################################################

    true_indexes = []
    for i in range(len(arr_mtx_t_epi)):
        # 1.0- Obtención de instantes característicos del ataque epiléptico
        mtx_inst = arr_mtx_t_epi[i]

        t_epi_start =   time2seg(time=mtx_inst[0],ref_time=mtx_t_reg[0])
        t_epi_end   =   time2seg(time=mtx_inst[1],ref_time=mtx_t_reg[0])

        interval = get_interval_range(t_epi_start, t_epi_end, fs, winlen)
        true_indexes = np.uint32(np.concatenate((true_indexes, interval)))

    ###########################################################################
    # PARTE 2 - Obtención de índices iniciales de intervalos falsos
    ###########################################################################

    # 2.0-  Obtención de instantes característicos del intervalo falso
    false_indexes = []
    for i in range(len(arr_mtx_t_epi) + 1):
        if i == 0:
            t_int_start = 0
            t_int_end = time2seg(time=arr_mtx_t_epi[0][0], ref_time=mtx_t_reg[0])

        elif i == len(arr_mtx_t_epi):
            t_int_start = time2seg(time=arr_mtx_t_epi[i - 1][1], ref_time=mtx_t_reg[0])
            t_int_end = time2seg(time=mtx_t_reg[1], ref_time=mtx_t_reg[0])

        else:
            t_int_start = time2seg(time=arr_mtx_t_epi[i - 1][1], ref_time=mtx_t_reg[0])
            t_int_end = time2seg(time=arr_mtx_t_epi[i][0], ref_time=mtx_t_reg[0])

        # 2.1- Obtención de índice inicial del intervalo falso
        interval = get_interval_range(t_int_start, t_int_end, fs, winlen)

        # 2.3- Concatenación del array de índices al array de índices iniciales de intervalos falsos
        false_indexes = np.uint32(np.concatenate((false_indexes, interval)))

    ###########################################################################
    # PARTE 3 - Obtención de feature vectors y labels de elementos verdaderos
    ###########################################################################

    # 3.0- Obtención de array con los segmentos de señal verdaderos
    arr_seg_sig_true = np.array([sig[idx:idx + step] for idx in true_indexes])
    labels_true = np.ones(len(true_indexes))

    # 3.1- Filtrado de los segmentos de la señal verdaderos
    arr_filtered_seg_sig_true = FIRfilterBP(arr_seg_sig_true)
    arr_filtered_seg_sig_true = IIRfilterBS(arr_filtered_seg_sig_true)

    ###########################################################################
    # PARTE 4 - Obtención de feature vectors y labels de elementos falsos
    ###########################################################################

    # 4.0- Cálculo de cantidades características
    n_true_segments = len(true_indexes)
    k_false_segments = np.uint16(n_true_segments / proportion + 1) - np.uint16(n_true_segments)

    # 4.1- Obtención de 'k' indices no repetidos de intervalos falsos
    selected_false_indexes = random.sample(population=list(false_indexes), k=k_false_segments)

    # 4.2- Obtención de array con los segmentos de la señal falsos
    arr_seg_sig_false = np.array([sig[idx: idx + step] for idx in selected_false_indexes])
    labels_false = np.zeros(len(selected_false_indexes))

    # 4.3- Filtrado de los segmentos de la señal falsos
    arr_filtered_seg_sig_false = FIRfilterBP(arr_seg_sig_false)
    arr_filtered_seg_sig_false = IIRfilterBS(arr_filtered_seg_sig_false)

    ###########################################################################
    # PARTE 5 - Unión de arrays
    ###########################################################################

    arr_seg = np.concatenate((arr_filtered_seg_sig_true, arr_filtered_seg_sig_false))
    arr_labels = np.concatenate((labels_true, labels_false))

    return arr_seg, arr_labels


def test():
    # Lectura
    DATA_DIR = r"../eeg_dataset/physionet.org/files/siena-scalp-eeg/1.0.0/"
    path01 = rf"{DATA_DIR}PN05/PN05-2.edf"
    raw01 = io.read_raw_edf(path01)
    info = raw01.info

    # Se obtienen los canales seleccionados del lazo izquierdo
    filt_ch_nms = ['EEG T3','EEG T5','EEG F7','EEG F3','EEG C3','EEG P3']

    # Seleccionar los datos de los canales filtrados
    raw01_filt = raw01.pick(filt_ch_nms)

    # Obtener los datos de los canales filtrados por nombre
    data_namefilt = raw01_filt.get_data()

    # Se convierten las mediciones a microvoltios
    data_namefilt = data_namefilt * 1e6

    # Matriz de inicio de registro y final de registro
    mtx_t_reg = np.array(["06.46.02", "09.19.47"])

    # Array de matrices con instantes iniciales y finales de ataque epileptico
    mtx_inst1 = np.array(["08.45.25", "08.46.00"])
    arr_mtx_t_epi = np.array([mtx_inst1])

    # Utilización de la función para una única señal
    arr_seg, arr_labels = getMeData(sig=data_namefilt[0], mtx_t_reg=mtx_t_reg, arr_mtx_t_epi=arr_mtx_t_epi)

    # Potencia
    fs = info["sfreq"]
    pot_seg = pot4signals(arr_seg, fs, divisor=1)
    pot_names = [f"potAbs{band.capitalize()}" for band in BANDAS.keys()]

    # Estadistica
    stats_data = stats_features(arr_seg)
    stats_names = list(stats_data.values())[:-1]
    stats_seg = stats_data["matriz de features stat"]

    # Feature fector
    arr_fv = np.hstack((pot_seg, stats_seg))
    columnas = pot_names + stats_names
    df_fv = pd.DataFrame(data=arr_fv, columns=columnas)
    print(df_fv.head(10))

    return  df_fv


if __name__ == '__main__':
    test()