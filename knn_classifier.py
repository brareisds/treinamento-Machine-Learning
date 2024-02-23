import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Importa a função tqdm

file = 'dados_hist.csv'


df = pd.read_csv(file, header=None)
df = df.rename(columns={256:'class'})

df_sample = df.sample(frac=0.05, random_state=1)


# Divide o conjunto de dados em duas matrizes: matriz X (bidimensional) com as caracteristicas 
# e matriz Y (unidimensional) com as classes de cada dado
# Dividir o conjunto de dados em treinamento e teste
X = df_sample.drop(df_sample.columns[256], axis=1)
Y = df_sample['class']

# Divide os dados entre conjunto de dados e de teste. Os conjuntos resultantes sao:
# X_treino (dados de treinamento), X_teste (dados de teste),
# y_treino (rótulos de treinamento) e y_teste (rótulos de teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.3, random_state=5)

# Calcula o min e max de cada feature
min_values = X_treino.min()
max_values = X_treino.max()

X_treino_normalizado = (X_treino - min_values) / (max_values - min_values)
X_teste_normalizado = (X_teste - min_values) / (max_values - min_values)

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(x1, x2):
    dist = np.sqrt(np.sum((x1 - x2) ** 2))
    return dist

def knn_predict(X_train, y_train, x_test, k=3):
    distances = []
    for _, x_train in X_train.iterrows():
        #print(x_train)
        # print(f'x_test = {x_test}')
        distance = euclidean_distance(x_test, x_train)
        distances.append(distance)

    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = np.take(y_train, k_indices)
    most_common = np.bincount(k_nearest_labels).argmax()
    return most_common

k = 3
correct_predictions = 0

# Fazendo a previsão para cada linha de teste
predictions = []
for index, row in tqdm(X_teste_normalizado.iterrows(), total=len(X_teste_normalizado)):
    prediction = knn_predict(X_treino_normalizado, y_treino, row, k)
    true_class = y_teste.loc[index]
    if prediction == true_class:
        correct_predictions += 1
    #print(f'prediction: {prediction} | classe verdadeira:', true_class)
    predictions.append(prediction)


# Imprimindo as previsões
# for i, prediction in enumerate(predictions):
#     print(f"Previsão para a linha {i}: {prediction}")

print(len(predictions))
print(f'correct predictions: {correct_predictions}')

accuracy = correct_predictions / len(predictions)
print(f'Acurácia: {accuracy}')
accuracy_percent = accuracy * 100
print(f'Acurácia: {accuracy_percent:.2f}%')


# Convertendo a série para uma lista
y_true = y_teste.tolist()

# Inicializando a matriz de confusão
confusion_matrix = [[0, 0], [0, 0]]

# Preenchendo a matriz de confusão
for i in range(len(y_true)):
    true_class = y_true[i]
    pred_class = predictions[i]
    confusion_matrix[int(true_class)][int(pred_class)] += 1

# Imprimindo a matriz de confusão formatada
print("Matriz de Confusão:")
print("           Empty     Occupied ")
print("---------------------------------------")
for i, row in enumerate(confusion_matrix):
    class_name = "Empty" if i == 0 else "Occupied"
    print(f" {class_name}  {row[0]:3}  {row[1]:9}")











