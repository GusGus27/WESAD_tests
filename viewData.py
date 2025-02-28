import pickle
import numpy as np

participantes = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

print("Cargando datos de los participantes...")

VentanasTotales = 0

for participante in participantes:

    # Cargar el objeto desde el archivo .pkl
    with open(f'data/WESAD/S{participante}/S{participante}.pkl', 'rb') as file:
        data_loaded = pickle.load(file, encoding='latin1')

    print("============= test subject " +str(participante)+ " ==================")

    print(data_loaded['signal']['wrist'].keys())

    llaves = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']
    llavesW = ['ACC', 'BVP', 'EDA', 'TEMP']

    for llave in llaves:
            
            print(data_loaded['signal']['chest'][llave].shape)


    for llave in llavesW:
    
        print(data_loaded['signal']['wrist'][llave].shape)

    total = data_loaded['label'].shape[0]

    estados = [1,2,3]

    total_valid = 0
    valid_windows = 0
    windowTotal = 0

    for estado in estados:

        count_ones = np.sum(data_loaded['label'] == estado)

        total_valid = total_valid + count_ones

        valid_windows = (count_ones - 21000)/350 + 1
        windowTotal = windowTotal + valid_windows
        print(f"Cantidad de datos iguales a {estado} en 'label': {count_ones}")
        print(f"Ventanas disponibles: {valid_windows}")
        print(f"Porcentaje de datos iguales a {estado} en 'label': {count_ones/total*100:.2f}%")


    print(f"Total de datos en validos: {windowTotal}")
    VentanasTotales = VentanasTotales + windowTotal

print(f"Total de datos en validos: {VentanasTotales}")