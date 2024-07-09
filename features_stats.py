import numpy as np
import matplotlib.pyplot as plt
from mne import io
import random
import scipy.signal as signal
import pandas as pd
import scipy.stats as stats

#se define rms, no encontre que Numpy o scipy la tengan
def rms(signal: np.array):
    return np.sqrt(np.mean(signal**2))

def stats_features(arr_signals : np.array,cant_parametros : int = 5):
    #se van a medir los parametros de kurtosis, RMS, skewness, media, desvio estandar para cada senal"

    #se crea un vector donde se almacenaran los datos de cada senal, 
    # la cantidad de filas de arr_signals es la cantidad de senales y can_parametros la cantidad de parametros

    stat_vector = np.zeros((arr_signals.shape[0],cant_parametros))
    for i in range(arr_signals.shape[0]):
        #se crea un vector para almacenar los features de una sola senal
        signal_features_vec = np.zeros((1,cant_parametros))

        # Kurtosis con Fisher=True
        signal_features_vec[0, 0] = stats.kurtosis(arr_signals[i], fisher=True)

        # RMS
        signal_features_vec[0, 1] = rms(arr_signals[i])

        # Skewness
        signal_features_vec[0, 2] = stats.skew(arr_signals[i])

        # Media
        signal_features_vec[0, 3] = np.mean(arr_signals[i])

        # Desviación estándar
        signal_features_vec[0, 4] = np.std(arr_signals[i])

        #paso los datos a la matriz que tiene todos los datos
        stat_vector[i] = signal_features_vec

    # se crea un diccionario en el cual se dice que informacion almacena cada columna del stat vector.    
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


