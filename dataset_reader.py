import re
from typing import Dict, List
from mne import io
import numpy as np

seizure_pattern = re.compile(r"Seizure n (\d+): ?\n?"
                            r"File name: ?([^\n]+)\n?"
                            r"Registration start time: ?([^\n]+)\n?"
                            r"Registration end time: ?([^\n]+)\n?"
                            r"Seizure start time: ?([^\n]+)\n?"
                            r"Seizure end time: ?([^\n]+)")

def get_seizure_events(filepath: str) -> List[Dict[str, str]]:
    """
    Encuentra la información sobre seizures en el archivo indicado
    """
    seizure_events = []

    # Leemos el archivo
    with open(filepath) as file:
        text = file.read()

    # Buscamos el patron
    matches = seizure_pattern.findall(text)
    for match in matches:
        # Lo pasamos a diccionario
        event_data = {
            "seizure_number": int(match[0]),
            "file_name": match[1].strip(),
            "registration_start_time": match[2].strip(),
            "registration_end_time": match[3].strip(),
            "seizure_start_time": match[4].strip(),
            "seizure_end_time": match[5].strip()
        }
        seizure_events.append(event_data)

    return seizure_events


def get_seizure_array(seizures: List[Dict[str, str]]) -> np.ndarray:
    """
    Transforma la información extraida por la función anterior a un array de eventos
    """
    return np.array([ [s["seizure_start_time"], s["seizure_end_time"]] for s in seizures])


def read_channel(filename: str, channel: str='EEG T3') -> np.ndarray:
    """
    Lee un canal de un archivo .edf
    :param filename: nombre el archivo a leer
    :param channel: nombre del canal a leer
    :return: la señal en dicho canal
    """
    raw = io.read_raw_edf(filename)
    ...


def main():
    DATA_DIR = r"../eeg_dataset/physionet.org/files/siena-scalp-eeg/1.0.0/"
    path = rf"{DATA_DIR}PN01/Seizures-list-PN01.txt"
    seizures = get_seizure_events(path)
    for seiz in seizures:
        print('\n'.join([f"{key}: {val}" for key, val in seiz.items()]))
        print()

    print(get_seizure_array(seizures))




if __name__ == '__main__':
    main()