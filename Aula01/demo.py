import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--input', help='Arquivo de entrada', required=True)
parser.add_argument('-k', '--kneighbors', help='Numero de vizinhos', type=int, default=1)
parser.add_argument('-p', '--plot', help='Se não deve plotar os graficos', action='store_false')

args = parser.parse_args()

df = pd.read_csv(args.input)

X_train, X_test, y_train, y_test = train_test_split(df[['x0', 'x1']], df['y'], test_size = 0.2)

clf = KNeighborsClassifier(n_neighbors=args.kneighbors)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f'Acuracia: {accuracy_score(y_test, y_pred)}')

if not args.plot:
    exit(0)

fig, ax = plt.subplots(1,3,figsize=(18,6))

ax[0].scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train, cmap=plt.cm.Spectral)
ax[0].scatter(X_test.iloc[:,0], X_test.iloc[:,1], marker = 'x', s=100)
ax[0].set_xlabel('Atributo 1')
ax[0].set_ylabel('Atributo 2')
ax[0].set_title('Exemplos de teste')

ax[1].scatter(X_train.iloc[:,0], X_train.iloc[:,1], c=y_train, cmap=plt.cm.Spectral)
ax[1].scatter(X_test.iloc[:,0], X_test.iloc[:,1], c=y_pred, marker = 'v', s=100, cmap=plt.cm.Spectral)
ax[1].set_xlabel('Atributo 1')
ax[1].set_ylabel('Atributo 2')
ax[1].set_title('Predições')


x1_min, x1_max = df.iloc[:, 0].min() - 1, df.iloc[:, 0].max() + 1
x2_min, x2_max = df.iloc[:, 1].min() - 1, df.iloc[:, 1].max() + 1

X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1), np.arange(x2_min, x2_max, 0.1))

Z = clf.predict(np.c_[X1.ravel(), X2.ravel()])
Z = Z.reshape(X1.shape)
    
ax[2].contourf(X1, X2, Z, cmap=plt.cm.Spectral, alpha=0.4)
ax[2].scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, s=40, cmap=plt.cm.Spectral)
ax[2].scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, s=40, marker='v', cmap=plt.cm.Spectral)
ax[2].set_xlabel('Atributo 1')
ax[2].set_ylabel('Atributo 2')
ax[2].set_title('Limites de decisão')

plt.show()
