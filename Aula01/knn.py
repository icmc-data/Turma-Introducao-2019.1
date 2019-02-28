import random
import math
from collections import Counter

def load_data(filename):
    """
    Carrega os dados de um arquivo passado como parametro
    """

    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(',')for x in content] 

    X = []
    y = []

    for c in content:
        c = [float(ci) for ci in c]
        X.append(c[:-1])
        y.append(c[-1])

    return X, y

def train_test_split(X, y):
    """
    Divide os dados entre conjunto de treino e teste (validação)
    """
    all_indexes = list(range(len(X)))
    train_idx = random.sample(all_indexes, int(0.2 * len(X)))
    test_idx = [i for i in all_indexes if i not in train_idx]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_test, y_test

def accuracy(y_test, y_pred):
    """
    Dadas as predições e as respostas esperadas calcula a acurácia
    """
    count = 0
    for y_test_i, y_pred_i in zip(y_test, y_pred):
        if y_test_i == y_pred_i:
            count += 1
    
    return count / len(y_test)

def argsort(vector):
    """
    Retorna os indices que iria ordenar o vetor
    (Exemplo: Input [4,1,5,2] -> Retorno [1,3,0,2])
    """
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(vector))]

def euclidian_distance(point_a, point_b):
    """
    Calcula a distancia euclidiana entre dois pontos
    """
    dist = 0
    for i in range(len(point_a)):
        dist += (point_a[i] - point_b[i]) ** 2

    dist = math.sqrt(dist)
    return dist    


def manhattan_distance(point_a, point_b):
    """
    Calcula a distancia manhattan entre dois pontos
    """
    dist = 0
    for i in range(len(point_a)):
        dist += abs(point_a[i] - point_b[i])
    return dist

def knn(X_train, y_train, X_test, k=1, dist_metric='euclidian'):
    y_pred = []
    
    # TODO: Implementar o algoritmo knn
    # Dicas: A função argsort e a classe Counter serão úteis

    return y_pred

if __name__ == '__main__':
    
    # Carregando os dados
    X, y = load_data('datasets/linear_hard.data')

    # Separando no nosso conjunto de validação
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    
    # TODO: Chamar a função knn e salvar o seu retorno na variavel y_pred
    y_pred = None

    #TODO: Verificar a acuracia do modelo passando os valores reais (y_test)
    # e os valores preditos (y_pred)
    acc = None

    print(f'Acuracia: {acc}')

    #TODO: Exercicio extra - Testar varios valores de k e ver qual funciona melhor
    # salvar os resulatados para cada k em um vetor e imprimir todas as acuracias obtidas
    # Dica: Pesquisar sobre grid search

