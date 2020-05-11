from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

mlp = MLPClassifier(max_iter=50)
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant'],
}


path = 'train2D.txt'
f_test = open(path, 'r')
data_test = np.genfromtxt(f_test)
f_test.close()
X = data_test[:, 1:]
y = data_test[:, 0]

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X, y)

print('Best parameters found:\n', clf.best_params_)
