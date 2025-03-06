import pickle
import numpy as np

participantes = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]

print("Cargando datos de los participantes...")

VentanasTotales = 0

with open(f'data/WESAD/S2/S2.pkl', 'rb') as file:
        data_loaded = pickle.load(file, encoding='latin1')

labels = data_loaded['label']  # (4255300,)
chest_signals = data_loaded['signal']['chest']  # Diccionario con las señales

# Crear una máscara para filtrar solo los datos con label 1, 2 o 3
valid_labels = {1, 2, 3}
mask = np.isin(labels, list(valid_labels))  # Array booleano con True en los índices a conservar

# Aplicar la máscara a las señales y a las etiquetas
filtered_labels = labels[mask]  # Filtrar etiquetas

filtered_signals = {}  # Diccionario para guardar señales filtradas
for key in chest_signals.keys():
    filtered_signals[key] = chest_signals[key][mask]  # Aplicar la máscara a cada señal

# Verificar los tamaños después del filtrado
print("Tamaño de las etiquetas filtradas:", filtered_labels.shape)
for key, signal in filtered_signals.items():
    print(f"Tamaño de {key} después del filtrado:", signal.shape)

print(filtered_signals['ECG'].shape)


'''for participante in participantes:

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

        valid_windows = (count_ones - 21000)/525 + 1
        windowTotal = windowTotal + valid_windows
        print(f"Cantidad de datos iguales a {estado} en 'label': {count_ones}")
        print(f"Ventanas disponibles: {valid_windows}")
        print(f"Porcentaje de datos iguales a {estado} en 'label': {count_ones/total*100:.2f}%")


    print(f"Total de datos en validos: {windowTotal}")
    VentanasTotales = VentanasTotales + windowTotal

print(f"Total de datos en validos: {VentanasTotales}")'''