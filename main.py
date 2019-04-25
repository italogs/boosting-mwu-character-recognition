from __future__ import print_function
import random
import numpy 
import time
import sys
GAMMA = 0.01
n_dimensions = 28

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


def MWU(matrix_data):
    p_distr = getInitialDistr()
    # best_result = 0
    # best_index = 0
    # O loop abaixo faz uma separacao horizontal. lado1 = superior, lado2 = inferior
    # A ideia e' iterar em todas as possiveis particoes horizontais
    # TO DO: Obter a melhor particao. Jogar este loop em uma funcao separada, ja que este codigo vai ser chamado dentro do loop Boosting MWU
    for i in range(0,n_dimensions+1):
            lado1 = []
            if(not(i == 0)):
                lado1 = matrix_data[:i,]
                # print("lado1",lado1)
                resultado_lado1 = most_common(lado1,p_distr)
                total_itens_lado1 = len(lado1) * len(lado1[0])
                erros_lado1 = total_itens_lado1 - resultado_lado1[1] 
                # print(resultado_lado1)
                if resultado_lado1[2] > 0.5 + GAMMA:
                    print('Weak-learner(lado1)',resultado_lado1[2])
            lado2 = []
            if(not(i == n_dimensions)):
                lado2 = matrix_data[i:,:]
                # print("lado2",lado2)
                resultado_lado2 = most_common(lado2,p_distr)
                total_itens_lado2 = len(lado2) * len(lado2[0])
                erros_lado2 = total_itens_lado2 - resultado_lado2[1] 
                # print(resultado_lado2)
                if resultado_lado2[2] > 0.5 + GAMMA:
                    print('Weak-learner(lado2)',resultado_lado2[2])

            
            

    # print("Melhor separador: ",best_index, best_result)
    
    # Loop do Boosting MWU
    # for it in range(0,T):
    #     print("Iteration: ",it)
    #     weak_learner = searchWeakLearner()
    #     print("Procurando weak-learner...")


if __name__ == "__main__":
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
    # Transformando os pontos de [0,255] em igual ou diferente
    list_class = []
    for i in range(1,len(number)):
        pixel_class = '='
        if a[i] != b[i]:
            pixel_class = '<>'
        list_class.append(pixel_class)

    #Transformando a lista acima em matriz
    matrix_data = []
    for i in range(0,len(list_class),n_dimensions):
        matrix_data.append(list_class[i:i+n_dimensions])
    matrix_data = numpy.array(matrix_data)

    # Exemplo para imprimir coluna 3 (nao apagar)
    # print(matrix_data[:,[3]])
    # Exemplo para imprimir linha 3 (nao apagar)
    # print(matrix_data[[3],: ])


    ## Fase 3 - Algoritmo MWU Boosting
    MWU(matrix_data)

