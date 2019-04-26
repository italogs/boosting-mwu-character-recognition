from __future__ import print_function
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import time
import sys
GAMMA = 0.01
n_dimensions = 28

def loadDataset(path):
    digits = []
    training_set_file = open(path, 'r')
    for line in training_set_file:
        digit = line.split(',')
        digits.append([ int(x) for x in digit ])
    return digits


def selectDistinctDigits(digits):
    a = None
    b = None
    for digit in digits:
        if a is None:
            a = [ int(x) for x in digit ]
        else:
            if digit[0] != a[0] and b is None:
                b = [ int(x) for x in digit ]
                break
    return a, b


def filterDigits(digits, digit_a, digit_b):
    digits_a = []
    digits_b = []
    for digit in digits:
        if digit[0] == digit_a[0]:
            digits_a.append(digit)
        elif digit[0] == digit_b[0]:
            digits_b.append(digit)
    return digits_a, digits_b


# Inicializa distribuicao de probabilidade
def getInitialDistr():
    initial_dist = 1.0/(n_dimensions*n_dimensions)
    p_distr = numpy.zeros(shape=(n_dimensions,n_dimensions))
    p_distr.fill(initial_dist)
    return p_distr

# Retorna o label mais comum, a sua frequencia e a sua probabilidade
def most_common(matrix_data,p_distr):
    freq_equal = 0
    p_equal = 0
    freq_diff = 0
    p_diff = 0
    for i in range(len(matrix_data)):
        for j in range(len(matrix_data[i])):
            if(matrix_data[i,j] == '='):
                freq_equal = freq_equal + 1
                p_equal = p_equal + p_distr[i,j]
            else:
                freq_diff = freq_diff + 1
                p_diff = p_diff + p_distr[i,j]
    return ['=',freq_equal,p_equal] if freq_equal >= freq_diff else ['!',freq_diff,p_diff]


# Provavelmente não é isso que precisamos.
def horizontalClassifier(matrix_data):
    p_distr = getInitialDistr()
    weak_learner = matrix_data[:1,]
    # best_result = 0
    # best_index = 0
    # O loop abaixo faz uma separacao horizontal. lado1 = superior, lado2 = inferior
    # A ideia e' iterar em todas as possiveis particoes horizontais
    # TO DO: Obter a melhor particao. Jogar este loop em uma funcao separada, ja que este codigo vai ser chamado dentro do loop Boosting MWU
    for i in range(0,n_dimensions+1):

        # Analyze an up partition containing all the first i * n_dimensions pixels.
        up_partition = []
        if(not(i == 0)):
            up_partition = matrix_data[:i,]
            down_partition = matrix_data[i:,]
            # print("up_partition",up_partition[0])
            up_partition_distr = most_common(up_partition, p_distr)
            up_partition_size = len(up_partition) * len(up_partition[0])
            up_partition_errors = up_partition_size - up_partition_distr[1]
            # print(resultado_lado1)
            if up_partition_distr[2] > 0.5 + GAMMA:
                print('Weak-learner(up_partition)', up_partition_distr[2])

        # Analyze a down partition containing all the last i * n_dimensions pixels.
        down_partition = []
        if(not(i == n_dimensions)):
            down_partition = matrix_data[i:,]
            # print("lado2",lado2)
            down_partition_distr = most_common(down_partition, p_distr)
            down_partition_size = len(down_partition) * len(down_partition[0])
            down_partition_errors = down_partition_size - down_partition_distr[1]
            # print(resultado_lado2)
            if down_partition_distr[2] > 0.5 + GAMMA:
                print('Weak-learner(down_partition)', down_partition_distr[2])


# Provavelmente não é isso que precisamos.
def verticalClassifier(matrix_data):
    p_distr = getInitialDistr()
    # best_result = 0
    # best_index = 0
    # O loop abaixo faz uma separacao vertical. lado1 = superior, lado2 = inferior
    # A ideia e' iterar em todas as possiveis particoes horizontais
    # TO DO: Obter a melhor particao. Jogar este loop em uma funcao separada, ja que este codigo vai ser chamado dentro do loop Boosting MWU
    for i in range(0,n_dimensions+1):

        # Analyze a left partition containing all the first l * n_dimensions column pixels.
        left_partition = []
        if(not(i == 0)):
            left_partition = matrix_data.transpose()[:i,]
            # print("left_partition",left_partition[0])
            left_partition_distr = most_common(left_partition, p_distr)
            left_partition_size = len(left_partition) * len(left_partition[0])
            left_partition_errors = left_partition_size - left_partition_distr[1]
            # print(resultado_lado1)
            if left_partition_distr[2] > 0.5 + GAMMA:
                print('Weak-learner(left_partition)', left_partition_distr[2])
        right_partition = []

        # Analyze a right partition containing all the last l * n_dimensions column pixels.
        right_partition = []
        if(not(i == n_dimensions)):
            right_partition = matrix_data.transpose()[i:,:]
            # print("lado2",lado2)
            right_partition_distr = most_common(right_partition, p_distr)
            right_partition_size = len(right_partition) * len(right_partition[0])
            right_partition_errors = right_partition_size - right_partition_distr[1]
            # print(resultado_lado2)
            if right_partition_distr[2] > 0.5 + GAMMA:
                print('Weak-learner(right_partition)', right_partition_distr[2])

def MWU(matrix_data):
    return 0




    # print("Melhor separador: ",best_index, best_result)

    # Loop do Boosting MWU
    # for it in range(0,T):
    #     print("Iteration: ",it)
    #     weak_learner = searchWeakLearner()
    #     print("Procurando weak-learner...")


if __name__ == "__main__":

    # Read the CSV dataset file.
    digits = loadDataset('mnist_test.csv')

    # Chooses the first two distinct digits from the read dataset.
    a = None
    b = None
    a, b = selectDistinctDigits(digits)

    print("\n\n Digit a = ", a[0])
    print("\n   Digit b = ", b[0])

    # Selects all digits that are either digit a or digit b.
    # Only these digits are relevant since we want to identify them.
    digits_a, digits_b = filterDigits(digits, a, b)

    print("\n\n Number of digits a = {} : {}".format(a[0], len(digits_a)))
    print("\n Number of digits b = {} : {}".format(b[0], len(digits_b)))


    # TODO: Repensar código abaixo.

    ## Fase 2 - Definir matriz de dados.
    # Transformando os pontos de [0,255] em igual ou diferente
    list_class = []
    for i in range(1, len(a)):
        pixel_class = '='
        if a[i] != b[i]:
            pixel_class = '<>'
        list_class.append(pixel_class)

    #Transformando a lista acima em matriz
    matrix_data = []
    for i in range(0,len(list_class),n_dimensions):
        matrix_data.append(list_class[i:i+n_dimensions])
    matrix_data = numpy.array(matrix_data)

    print("\n\n---------------\n\n")
    print(matrix_data[:,])
    print("\n\n\n")
    print(matrix_data.transpose()[:5,])
    print("\n\nmatrix_data = \n\n\t", matrix_data)
    # Exemplo para imprimir coluna 3 (nao apagar)
    # print(matrix_data[:,[3]])
    # Exemplo para imprimir linha 3 (nao apagar)
    # print(matrix_data[[3],: ])


    ## Fase 3 - Algoritmo MWU Boosting
horizontalClassifier(matrix_data)
