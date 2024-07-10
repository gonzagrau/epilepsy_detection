import numpy as np
import scipy.stats as stats
from typing import Dict

def rms(sig: np.ndarray, **kwargs) -> np.ndarray:
    """
    root-mean-square of a numpy array (vectorized by the "mean" of it)
    """
    return np.sqrt(np.mean(sig**2, **kwargs))

def stats_features(arr_signals : np.ndarray) -> Dict:
    # se van a medir los parametros de kurtosis, RMS, skewness, media, desvio estandar para cada senal"
    # se crea un vector donde se almacenaran los datos de cada senal,
    # la cantidad de filas de arr_signals es la cantidad de senales y can_parametros la cantidad de parametros
    cant_parametros = 5
    stat_vector = np.zeros((arr_signals.shape[0], cant_parametros))
    stat_vector[:, 0] = stats.kurtosis(arr_signals, fisher=True, axis=1)
    stat_vector[:, 1] = rms(arr_signals, axis=1)
    stat_vector[:, 2] = stats.skew(arr_signals, axis=1)
    stat_vector[:, 3] = np.mean(arr_signals, axis=1)
    stat_vector[:, 4] = np.std(arr_signals, axis=1)

    stat_dic = {
        0: "kurtosis",
        1: "RMS",
        2: "skewness",
        3: "media",
        4: "desvio estandar",
        "matriz de features stat": stat_vector
    }
    return stat_dic

def main():
    # Generar una señal de prueba (por ejemplo, señal sinusoidal)
    fs = 1000  # Frecuencia de muestreo (Hz)
    t = np.arange(0, 1, 1/fs)  # Vector de tiempo de 0 a 1 segundo
    vector_senales = []
    for i in range(5):
        f = np.random.randint(low=4, high=15)
        signal = np.sin(2 * np.pi * f * t)  # Señal sinusoidal
        vector_senales.append(signal)

    vector_senales = np.array(vector_senales)
    dic = stats_features(vector_senales)   
    print(dic)

if __name__ == '__main__':
    main()


