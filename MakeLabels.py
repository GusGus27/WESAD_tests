import os
import pandas as pd
from tqdm import tqdm
import numpy as np

DATA_DIR = r'FAIR_Data\Hombres\Control'
OUTPUT_DIR = r'dataProcessed'

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(DATA_DIR) if not f.endswith('_markers.csv') and f.endswith('.csv')]

for file in tqdm(files, desc="Procesando archivos"):
    print(f"Procesando: {file}")
    try:
        data_file = os.path.join(DATA_DIR, file)
        markers_file = os.path.join(DATA_DIR, file.replace('.csv', '_markers.csv'))
        output_file = os.path.join(OUTPUT_DIR, file)

        # Cargar marcadores
        conditions = pd.read_csv(markers_file, index_col=None)
        print(conditions)

        # Lista para almacenar los chunks procesados
        chunk_list = []
        first_row = True  # Para controlar la eliminación de la primera fila

        for chunk in pd.read_csv(data_file, index_col=None, header=11, chunksize=50000):
            chunk.dropna(axis=1, how='all', inplace=True)  # Eliminar columnas vacías

            if first_row:
                chunk.drop(index=chunk.index[0], inplace=True)  # Eliminar primera fila
                first_row = False  # Solo eliminar en el primer chunk

            # Crear columna de tiempo
            chunk['time'] = chunk['hrs'] * 3600
            chunk['time'] = chunk['time'].round(3)
            chunk = chunk[['time'] + [col for col in chunk.columns if col != 'time']]
            chunk.drop(columns=['hrs'], inplace=True)

            # Función para asignar etiquetas
            def asignar_label_vectorized(df, conditions):
                cond1 = (df['time'] > conditions.iloc[0]['time']) & (df['time'] <= conditions.iloc[2]['time'])
                cond2 = (df['time'] > conditions.iloc[3]['time']) & (df['time'] <= conditions.iloc[4]['time'])
                cond3 = ((df['time'] > conditions.iloc[5]['time']) & (df['time'] <= conditions.iloc[7]['time']))
                cond4 = ((df['time'] > conditions.iloc[7]['time']) & (df['time'] <= conditions.iloc[9]['time']))

                df['label'] = np.select([cond1, cond2, cond3, cond4], [1, 2, 3, 4], default=0)
                return df
            
            def asignar_label_vectorized_control(df, conditions):
                cond1 = (df['time'] > conditions.iloc[0]['time']) & (df['time'] <= conditions.iloc[2]['time'])
                cond2 = (df['time'] > conditions.iloc[3]['time']) & (df['time'] <= conditions.iloc[4]['time']) | ((df['time'] > conditions.iloc[7]['time']) & (df['time'] <= conditions.iloc[9]['time']))
                cond3 = ((df['time'] > conditions.iloc[5]['time']) & (df['time'] <= conditions.iloc[6]['time']))

                df['label'] = np.select([cond1, cond2, cond3], [1, 2, 5], default=0)
                return df

            chunk = asignar_label_vectorized_control(chunk, conditions)
            chunk_list.append(chunk)

        # Guardar el DataFrame final
        final_df = pd.concat(chunk_list, ignore_index=True)

        final_df.rename(columns={"CH1": "RESP", "CH2": "PPG", "CH13": "ECG", "CH14" : "EDA"}, inplace=True)

        final_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error procesando {data_file}: {e}")