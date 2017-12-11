import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin

class MySGDClassifier(BaseEstimator, ClassifierMixin):
    """
    Предполагается, что в выборке всегда 2 класса
    """
    
    def __init__(self, C=1, alpha=0.01, max_epoch=10):
        """
        C - коэф. регуляризации
        alpha - скорость спуска
        max_epoch - максимальное количество эпох
        """
        
        self.C = C
        self.alpha = alpha
        self.max_epoch = max_epoch
    
    def fit(self, X, y=None):
        '''
        Обучение модели
        '''
        
        Y = np.ones((X.shape[0], X.shape[1] + 1))
        Y[:, :-1] = X
        
        self.weights = np.zeros((Y.shape[1]))
        
        epoches = 0
        
        while True:
            epoches += 1
            alpha = self.alpha / epoches

            indicies = np.arange(X.shape[0])
            np.random.shuffle(indicies)
            batches = np.array_split(indicies, X.shape[0] / 32 + 1)

            for b in batches:
                weights = self.weights
                self.weights = weights - alpha*self.grad(X[b], y[b])

            if epoches == self.epoches or np.linalg.norm(weights - self.weights) < 0.01:
                break
        
        return self
    
    def partial_fit(self, X, y=None):
        '''
        По желанию - метод дообучения модели на новых данных
        '''
    
        return self
        
    def predict(self, X):
        '''
        Возвращение метки класса
        '''
        
        return y_hat
    
    def predict_proba(self, X):
        '''
        Возвращение вероятности каждого из классов
        '''

        
        return y_hat_proba
    
    def grad(self, X, y):
        '''
        Возвращает градиент X
        '''

        margin = X.dot(self.weights) * y
        
        grad = 1/(1 + np.exp(margin))
        grad *= -y
        grad = X.T.dot(grad) + self.weights * (2/self.C)
        
        return grad

array = np.array([[1, 2, 1], [1, 2, 1]])
y = np.array([1, -1])
sgd = MySGDClassifier()
sgd.fit(array, y)