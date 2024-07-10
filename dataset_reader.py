import re
from typing import Dict, List

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
            "file_name": match[1],
            "registration_start_time": match[2],
            "registration_end_time": match[3],
            "seizure_start_time": match[4],
            "seizure_end_time": match[5]
        }
        seizure_events.append(event_data)

    return seizure_events


def get_seizure_array(seizures: List[Dict[str, str]]) -> np.ndarray:
    """
    Transforma la información extraida por la función anterior a un array de eventos
    """
    return np.array([ [s["seizure_start_time"], s["seizure_end_time"]] for s in seizures])

def main():
    DATA_DIR = r"../eeg_dataset/physionet.org/files/siena-scalp-eeg/1.0.0/"
    path = rf"{DATA_DIR}PN05/Seizures-list-PN05.txt"
    seizures = get_seizure_events(path)
    for seiz in seizures:
        print('\n'.join([f"{key}: {val}" for key, val in seiz.items()]))
        print()

    print(get_seizure_array(seizures))




if __name__ == '__main__':
    main()