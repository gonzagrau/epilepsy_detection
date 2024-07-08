import numpy as np
import matplotlib.pyplot as plt
from mne import io
import random
import scipy.signal as signal
import pandas as pd

# CONTANTE: Bandas de ondas "del alfabeto griego"
BANDAS = np.array([[0.5, 4],
                   [4,8],
                   [8,13],
                   [13,30],
                   [30,50]])



def time2seg(hora,minuto,segundo):
    seg1 = hora*3600
    seg2 = minuto*60
    seg3 = segundo

    seg = seg1+seg2+seg3

    return seg


def FIRfilterBP(arr_signals, freq=[1, 100], order=(2 ** 8) + 1, fs=512):
    # 0- Cálculo de los coeficientes del filtro FIR
    b_bp = signal.firwin(numtaps=order, cutoff=freq, window="blackman", pass_zero="bandpass", fs=fs)

    # 1- Aplicación del filtro a las señales
    arr_filtered_signals = np.array([signal.filtfilt(b_bp, 1, sig) for sig in arr_signals])

    return arr_filtered_signals


def IIRfilterBS(arr_signals, freq=[49.5, 50.5], order=2, fs=512):
    # 0- Cálculo de los coeficientes del filtro IIR
    b_bs, a_bs = signal.iirfilter(N=order, Wn=freq, btype="bandstop", ftype='butter', fs=fs, output='ba')

    # 1- Aplicación del filtro a las señales
    arr_filtered_signals = np.array([signal.filtfilt(b_bs, a_bs, sig) for sig in arr_signals])

    return arr_filtered_signals


def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(K-lst[i]))]


def pot4band(arr_freq, psd):
    # 0- Definición de lista de almacenamiento
    arr_pot_abs4band = []

    # 1- Cálculo de potencias absolutas y relativas para cada banda
    for banda in BANDAS:
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
    totalPow = np.sum(arr_pot_abs4band)
    arr_pot_rel4band = (100 * arr_pot_abs4band) / totalPow

    # 6- Concatenación de arrays
    arr_storage = np.array([totalPow])
    arr_storage = np.concatenate((arr_storage, arr_pot_abs4band))
    arr_storage = np.concatenate((arr_storage, arr_pot_rel4band))

    return arr_storage


def pot4signals(arr_signals, fs=512, divisor=100):
    # 0- Cálculo del nper
    nper = int(len(arr_signals[0]) // divisor)

    # 1- Cálculo del array de PSDs para cada señal del array de señales
    arr_freq = signal.welch(x=arr_signals[0], fs=fs, noverlap=nper / 2, nperseg=nper)[0]
    arr_psd = np.array([signal.welch(x=sig, fs=fs, noverlap=nper / 2, nperseg=nper)[1] for sig in arr_signals])

    """plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(arr_signals[0])), arr_signals[0])
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(arr_freq, arr_psd[0])
    plt.xlim(left=0,right=50)
    plt.show()"""

    # 2 Cálculo de potencias para cada señal del array de señales
    arr_pot4signals = np.array([pot4band(arr_freq, psd) for psd in arr_psd])

    return arr_pot4signals


def getMeData(signal, mtx_t_reg, arr_mtx_t_epi, winlen=2, fs=512, proportion=0.4):
    # 0- Declaración de parámetros importantes
    step = fs * winlen

    ###########################################################################
    # PARTE 1 - Obtención de instante inicial y final
    ###########################################################################

    t_reg_start = time2seg(mtx_t_reg[0][0], mtx_t_reg[0][1], mtx_t_reg[0][2])
    t_reg_end = time2seg(mtx_t_reg[1][0], mtx_t_reg[1][1], mtx_t_reg[1][2])

    ###########################################################################
    # PARTE 2 - Obtención de índices iniciales de intervalos verdaderos
    ###########################################################################

    true_indexes = []
    for i in range(len(arr_mtx_t_epi)):
        # 2.0- Obtención de instantes característicos del ataque epiléptico
        mtx_inst = arr_mtx_t_epi[i]

        t_epi_start = time2seg(mtx_inst[0][0], mtx_inst[0][1], mtx_inst[0][2]) - t_reg_start
        t_epi_end = time2seg(mtx_inst[1][0], mtx_inst[1][1], mtx_inst[1][2]) - t_reg_start

        # 2.1- Obtención de índice inicial del ataque epiléptico
        i_epi_start = fs * t_epi_start

        # 2.2- Obtención de índices iniciales de cada segmento de longitud 'step'
        start = i_epi_start
        stop = step * ((t_epi_end - t_epi_start) // winlen) + i_epi_start
        interval = np.arange(start=start, stop=stop, step=step)

        # 2.3- Concatenación del array de índices al array de índices iniciales de intervalos verdaderos
        true_indexes = np.uint32(np.concatenate((true_indexes, interval)))

    ###########################################################################
    # PARTE 3 - Obtención de índices iniciales de intervalos falsos
    ###########################################################################

    false_indexes = []
    for i in range(len(arr_mtx_t_epi) + 1):
        if i == 0:
            # 3.0-  Obtención de instantes característicos del intervalo falso
            t_int_start = t_reg_start - t_reg_start
            t_int_end = time2seg(arr_mtx_t_epi[0][0][0], arr_mtx_t_epi[0][0][1], arr_mtx_t_epi[0][0][2]) - t_reg_start

        elif i == len(arr_mtx_t_epi):
            # 3.0-  Obtención de instantes característicos del intervalo falso
            t_int_start = time2seg(arr_mtx_t_epi[i - 1][1][0], arr_mtx_t_epi[i - 1][1][1],
                                   arr_mtx_t_epi[i - 1][1][2]) - t_reg_start
            t_int_end = t_reg_end - t_reg_start

        else:
            # 3.0-  Obtención de instantes característicos del intervalo falso
            t_int_start = time2seg(arr_mtx_t_epi[i - 1][1][0], arr_mtx_t_epi[i - 1][1][1],
                                   arr_mtx_t_epi[i - 1][1][2]) - t_reg_start
            t_int_end = time2seg(arr_mtx_t_epi[i][0][0], arr_mtx_t_epi[i][0][1], arr_mtx_t_epi[i][0][2]) - t_reg_start

        # 3.1- Obtención de índice inicial del intervalo falso
        i_int_start = fs * t_int_start

        # 3.2- Obtención de índices iniciales de cada segmento de longitud 'step'
        start = i_int_start
        stop = step * ((t_int_end - t_int_start) // winlen) + i_int_start
        interval = np.arange(start=start, stop=stop, step=step)

        # 3.3- Concatenación del array de índices al array de índices iniciales de intervalos falsos
        false_indexes = np.uint32(np.concatenate((false_indexes, interval)))

    ###########################################################################
    # PARTE 4 - Obtención de feature vectors y labels de elementos verdaderos
    ###########################################################################

    # 4.0- Obtención de array con los segmentos de señal verdaderos
    arr_seg_sig_true = np.array([signal[true_indexes[i]:(true_indexes[i] + step)] for i in range(len(true_indexes))])

    # 4.1- Filtrado de los segmentos de la señal verdaderos
    arr_filtered_seg_sig_true = FIRfilterBP(arr_seg_sig_true)
    arr_filtered_seg_sig_true = IIRfilterBS(arr_filtered_seg_sig_true)

    # 4.2- Potencias de los segmentos de la señal verdaderos
    arr_pot4segsig_true = pot4signals(arr_filtered_seg_sig_true, divisor=1)

    # 4.OTROS FEATURES

    # 4.FINAL-
    # fv_true = []
    fv_true = arr_pot4segsig_true
    labels_true = np.ones(len(true_indexes))

    ###########################################################################
    # PARTE 5 - Obtención de feature vectors y labels de elementos falsos
    ###########################################################################

    # 5.0- Cálculo de cantidades características
    n_true_segments = len(true_indexes)
    k_false_segments = np.uint16(n_true_segments / proportion + 1) - np.uint16(n_true_segments)

    # 5.1- Obtención de 'k' indices no repetidos de intervalos falsos
    selected_false_indexes = random.sample(population=list(false_indexes), k=k_false_segments)

    # 5.2- Obtención de array con los segmentos de la señal falsos
    arr_seg_sig_false = np.array([signal[selected_false_indexes[i]:(selected_false_indexes[i] + step)] for i in
                                  range(len(selected_false_indexes))])

    # 5.3- Filtrado de los segmentos de la señal falsos
    arr_filtered_seg_sig_false = FIRfilterBP(arr_seg_sig_false)
    arr_filtered_seg_sig_false = IIRfilterBS(arr_filtered_seg_sig_false)

    # 5.4- Potencias de los segmentos de la señal falsos
    arr_pot4segsig_false = pot4signals(arr_filtered_seg_sig_false, divisor=1)

    # 5.OTROS FEATURES

    # 5.FINAL-
    # fv_false = []
    fv_false = arr_pot4segsig_false
    labels_false = np.zeros(len(selected_false_indexes))

    ###########################################################################
    # PARTE 6 - Unión de arrays
    ###########################################################################

    arr_fv = np.concatenate((fv_false, fv_true))
    arr_labels = np.concatenate((labels_false, labels_true))

    return arr_fv, arr_labels


def test():
    # Lectura
    DATA_DIR = r"../eeg_dataset/physionet.org/files/siena-scalp-eeg/1.0.0/"
    path01 = rf"{DATA_DIR}PN05/PN05-2.edf"
    raw01 = io.read_raw_edf(path01)
    paciente = "PN05"
    realizacion = "2"

    ###########################################################################
    # Organización de la información dentro del archivo .edf
    ###########################################################################

    info = raw01.info

    print(info)
    print(info['ch_names'])

    ###########################################################################
    # Obtención de las mediciones a utilizar y otros datos importantes
    ###########################################################################

    # Se obtiene el nombre de todos los canales
    ch_nms = info["ch_names"]

    # Se obtienen los canales seleccionados del lazo izquierdo
    #filt_ch_nms = ['EEG T3', 'EEG T5']
    filt_ch_nms = ['EEG T3','EEG T5','EEG F7','EEG F3','EEG C3','EEG P3']

    # Seleccionar los datos de los canales filtrados
    raw01_filt = raw01.pick(filt_ch_nms)

    # Obtener los datos de los canales filtrados por nombre
    data_namefilt = raw01_filt.get_data()

    # Se convierten las mediciones a microvoltios
    data_namefilt = data_namefilt * 1e6

    # Dimensiones de 'data_filt'
    dim_data_filt = np.shape(data_namefilt)

    # Verifición
    print("Canales filtrados:", filt_ch_nms)
    print(f"Cantidad de canales resultantes: {dim_data_filt[0]}")
    print(f"Cantidad de datos en cada canal: {dim_data_filt[1]}")

    # Frecuencia de muestreo en Hz
    fs = info["sfreq"]

    # Cantidad de muestras tomadas
    len_data = dim_data_filt[1]

    # Array de instantes
    start = 0
    stop = (1 / fs) * len_data
    arr_t = np.arange(start=start, stop=stop, step=(1 / fs))

    # Verificación de datos
    print("Frecuencia de muestreo:", fs)
    print("Instantes [s]:", arr_t[0], "...", arr_t[-1])

    # Matriz de inicio de registro y final de registro
    mtx_t_reg = np.array([[6, 46, 2], [9, 19, 47]])

    # Array de matrices con instantes iniciales y finales de ataque epileptico
    mtx_inst1 = np.array([[8, 45, 25], [8, 46, 0]])
    arr_mtx_t_epi = np.array([mtx_inst1])

    # Utilización de la función para una única señal
    arr_fv, arr_labels = getMeData(signal=data_namefilt[0], mtx_t_reg=mtx_t_reg, arr_mtx_t_epi=arr_mtx_t_epi)
    columnas = ['Pot total', 'potAbsDelta', 'potAbsTheta', 'potAbsAlpha', 'potAbsBeta', 'potAbsGamma', 'potRelDelta',
                'potRelTheta', 'potRelAlpha', 'potRelBeta', 'potRelGamma']
    df_fv = pd.DataFrame(data=arr_fv, columns=columnas)
    print(df_fv.head(10))
    # Feature vector:
    # [Pot total, potAbsDelta, potAbsTheta, potAbsAlpha, potAbsBeta, potAbsGamma, potRelDelta, potRelTheta, potRelAlpha, potRelBeta, potRelGamma]


if __name__ == '__main__':
    test()