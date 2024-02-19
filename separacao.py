import pandas as pd
import numpy as np

# Leia o arquivo CSV original
df = pd.read_csv('dados_hist.csv')

# Calcule o número de linhas para 70% e 30%
total_linhas = len(df)
linhas_70_porcento = int(total_linhas * 0.7)
linhas_30_porcento = total_linhas - linhas_70_porcento

# Divida as linhas em dois dataframes
df_70_porcento = df.iloc[:linhas_70_porcento]
df_30_porcento = df.iloc[linhas_70_porcento:]

# Salve os dataframes em arquivos CSV
df_70_porcento.to_csv('treinamento.csv', index=False)
df_30_porcento.to_csv('teste.csv', index=False)

def Euclidean_distance(p, q):
    dist = np.sqrt(np.sum(np.square(p-q)))
    return dist

