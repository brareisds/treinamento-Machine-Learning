import csv

# Caminho para o arquivo CSV
arquivo_csv = 'train.csv'

# Abre o arquivo CSV em modo de leitura
with open(arquivo_csv, newline='') as csvfile:
    # Cria um leitor CSV
    reader = csv.reader(csvfile, delimiter=',')
    # Lê a primeira linha para determinar o número de colunas
    primeira_linha = next(reader)
    numero_colunas = len(primeira_linha)

print(f'O arquivo possui {numero_colunas} colunas.')
