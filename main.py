from __future__ import print_function
import random
import numpy as np

## Fase 1 - Ler arquivo
a = None
b = None
training_set_file = open('data/mnist_test.csv', 'r')
for line in training_set_file:
    number = line.split(',')
    if a is None:
        a = [ int(x) for x in number ]
    else:    
        if number[0] != a[0] and b is None:
            b = [ int(x) for x in number ]
            break


## Fase 2 - Definir matriz de dados.
print('a:',a[0])
print('b:', b[0])

# Transformando os pontos de [0,255] em igual ou diferente
list_class = []
for i in range(1,len(number)):
    pixel_class = '='
    if a[i] != b[i]:
        pixel_class = '<>'
    list_class.append(pixel_class)

#Transformando a lista acima em matriz
matrix_data = []
for i in range(0,len(list_class),28):
    matrix_data.append(list_class[i:i+28])
matrix_data = np.matrix(matrix_data)

# Exemplo para imprimir coluna 3 (nao apagar)
# print(matrix_data[:,[3]])
# Exemplo para imprimir linha 3 (nao apagar)
# print(matrix_data[[3],: ])


## Fase 3 - Algoritmo MWU Boosting
# Realizando a distribuicao de probabilidade 1/28*28
p = [ 1.0/len(list_class) for x in range(len(list_class)) ]
