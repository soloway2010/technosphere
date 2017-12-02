# coding=latin-1

from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import optimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

class MyDecisionTreeClassifier:
    NON_LEAF_TYPE = 0
    LEAF_TYPE = 1

    def __init__(self, min_samples_split=2, max_depth=None, sufficient_share=1.0, criterion='gini', max_features=None):
        self.tree = dict()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.sufficient_share = sufficient_share
        self.num_class = -1
        if criterion == 'gini':
            self.G_function = self.__gini
        elif criterion == 'entropy':
            self.G_function = self.__entropy
        elif criterion == 'misclass':
            self.G_function = self.__misclass
        else:
            print 'invalid criterion name'
            raise

        if max_features == 'sqrt':
            self.get_feature_ids = self.__get_feature_ids_sqrt
        elif max_features == 'log2':
            self.get_feature_ids = self.__get_feature_ids_log2
        elif max_features == None:
            self.get_feature_ids = self.__get_feature_ids_N
        else:
            print 'invalid max_features name'
            raise

    def __gini(self, l_c, l_s, r_c, r_s):
        l_s = l_s.astype('float')
        r_s = r_s.astype('float')
        
        l_c = l_c / l_s
        r_c = r_c / r_s

        I_SL = np.square(l_c)
        I_SL = (1 - np.sum(I_SL, axis=1)).reshape(-1, 1)

        I_SR = np.square(r_c)
        I_SR = (1 - np.sum(I_SR, axis=1)).reshape(-1, 1)

        return ((l_s*I_SL + r_s*I_SR)/(l_s + r_s)).reshape(-1)
    
    def __entropy(self, l_c, l_s, r_c, r_s):
    	l_s = l_s.astype('float')
        r_s = r_s.astype('float')

    	l_c = l_c / l_s
        r_c = r_c / r_s

      	np.place(l_c, l_c == 0, 1)
        I_SL = l_c * np.log(l_c)
        I_SL = (-np.sum(I_SL, axis=1)).reshape(-1, 1)

        np.place(r_c, r_c == 0, 1)
        I_SR = r_c * np.log(r_c)
        I_SR = (-np.sum(I_SR, axis=1)).reshape(-1, 1)

        return ((l_s*I_SL + r_s*I_SR)/(l_s + r_s)).reshape(-1)

    def __misclass(self, l_c, l_s, r_c, r_s):
    	l_s = l_s.astype('float')
        r_s = r_s.astype('float')

    	l_c = l_c / l_s
        r_c = r_c / r_s

        I_SL = (1 - np.max(l_c, axis=1)).reshape(-1, 1)

        I_SR = (1 - np.max(r_c, axis=1)).reshape(-1, 1)

        return ((l_s*I_SL + r_s*I_SR)/(l_s + r_s)).reshape(-1)

    def __get_feature_ids_sqrt(self, n_feature):
        feature_ids = range(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[:int(np.sqrt(n_feature))]
        
    def __get_feature_ids_log2(self, n_feature):
        feature_ids = range(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[:int(np.log2(n_feature))]

    def __get_feature_ids_N(self, n_feature):
        return range(n_feature)
    
    def __sort_samples(self, x, y):
        sorted_idx = x.argsort()
        return x[sorted_idx], y[sorted_idx]

    def __div_samples(self, x, y, feature_id, threshold):
        left_mask = x[:, feature_id] > threshold
        right_mask = ~left_mask
        return x[left_mask], x[right_mask], y[left_mask], y[right_mask]

    def __find_threshold(self, x, y):
        # Что делает этот блок кода?
        # Заносит в sorted_x и sorted_y отсортированные значения фичи и их метки
        # В class_number заносит количество уникальных меток
        sorted_x, sorted_y = self.__sort_samples(x, y)
        class_number = np.unique(y).shape[0]
        
        # Что делает этот блок кода?
        # Ищет все возможные места разделения значений на две части
        splitted_sorted_y = sorted_y[self.min_samples_split:-self.min_samples_split]
        r_border_ids = np.where(splitted_sorted_y[:-1] != splitted_sorted_y[1:])[0] + (self.min_samples_split + 1)
        
        if len(r_border_ids) == 0:
            return float('+inf'), None
    
        # Что делает этот блок кода?
        # Магия
        eq_el_count = r_border_ids - np.append([self.min_samples_split], r_border_ids[:-1])
        one_hot_code = np.zeros((r_border_ids.shape[0], class_number))
        one_hot_code[np.arange(r_border_ids.shape[0]), sorted_y[r_border_ids - 1]] = 1
        class_increments = one_hot_code * eq_el_count.reshape(-1, 1)
        class_increments[0] = class_increments[0] + np.bincount(y[:self.min_samples_split], minlength=class_number)
        
        # Что делает этот блок кода?
        # Вычисляет матрицу, где каждая строчка это количество значений в каждом классе
        # Заносит в l_class_count матрицу значений слева, и в r_class_count справа
        # Заносит в l_sizes и r_sizes общее количетсво объектов справа и слева для каждого варианта разбиения
        l_class_count = np.cumsum(class_increments, axis=0)        
        r_class_count = np.bincount(y) - l_class_count
        l_sizes = r_border_ids.reshape(l_class_count.shape[0], 1)
        r_sizes = sorted_y.shape[0] - l_sizes

        # Что делает этот блок кода?
        # Для кажого варианта разбиения подсчитывет его качество и выбирает наилучший вариант
        gs = self.G_function(l_class_count, l_sizes, r_class_count, r_sizes)
        idx = np.argmin(gs)
    
        # Что делает этот блок кода?
        # Заносит в left_el_id количество значений слева для лучшего разбиения
        # Возвращает оценку лучшего разбиения и значение по которому нужно разбивать
        left_el_id = l_sizes[idx][0]
        return gs[idx], (sorted_x[left_el_id-1] + sorted_x[left_el_id]) / 2.0

    def __fit_node(self, x, y, node_id, depth, pred_f=-1):
        # Ваш код
        # Необходимо использовать следующее:
        # self.LEAF_TYPE
        # self.NON_LEAF_TYPE

        # self.tree
        # self.max_depth
        # self.sufficient_share
        # self.min_samples_split

        # self.get_feature_ids
        # self.__find_threshold
        # self.__div_samples
        # self.__fit_node
        
        min_coef = float('+inf')
        cor_threshold = None
        cor_feature_id = None

        feature_ids = self.get_feature_ids(self.num_class)

        for i in feature_ids:
        	coef, threshold = self.__find_threshold(x[:, i], y)
        	if coef < min_coef:
        		min_coef = coef
        		cor_threshold = threshold
        		cor_feature_id = i

        if len(y) < self.min_samples_split or depth == self.max_depth or cor_threshold == None :
        	class_count = np.bincount(y)
        	class_count = np.argsort(class_count)
        	self.tree[node_id] = (self.LEAF_TYPE, class_count[-1], 0)
        	return

        if self.sufficient_share < 1:
        	class_count = np.bincount(y)
        	class_count_ids = np.argsort(class_count)
        	if class_count[class_count_ids[-1]]/len(class_count) >= self.sufficient_share:
        		self.tree[node_id] = (self.LEAF_TYPE, class_count_ids[-1], 0)
        		return

        x_l, x_r, y_l, y_r = self.__div_samples(x, y, cor_feature_id, cor_threshold)

        if x_r.shape[0] == x.shape[0] or x_l.shape[0] == x.shape[0]:
        	class_count = np.bincount(y)
        	class_count = np.argsort(class_count)
        	self.tree[node_id] = (self.LEAF_TYPE, class_count[-1], 0)
        	return

        self.tree[node_id] = (self.NON_LEAF_TYPE, cor_feature_id, cor_threshold)
        self.__fit_node(x_r, y_r, 2*node_id + 1, depth + 1)
        self.__fit_node(x_l, y_l, 2*node_id + 2, depth + 1)

    
    def fit(self, x, y):
        self.num_class = np.unique(y).size
        self.__fit_node(x, y, 0, 0) 

    def __predict_class(self, x, node_id):
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_id, threshold = node
            if x[feature_id] > threshold:
                return self.__predict_class(x, 2 * node_id + 1)
            else:
                return self.__predict_class(x, 2 * node_id + 2)
        else:
            return node[1]

    def __predict_probs(self, x, node_id):
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_id, threshold = node
            if x[feature_id] > threshold:
                return self.__predict_probs(x, 2 * node_id + 1)
            else:
                return self.__predict_probs(x, 2 * node_id + 2)
        else:
            return node[2]
        
    def predict(self, X):
        return np.array([self.__predict_class(x, 0) for x in X])
    
    def predict_probs(self, X):
        return np.array([self.__predict_probs(x, 0) for x in X])

    def fit_predict(self, x_train, y_train, predicted_x):
        self.fit(x_train, y_train)
        return self.predict(predicted_x)

df = pd.read_csv('./cs-training.csv', sep=',').dropna()

x = df.as_matrix(columns=df.columns[1:])
y = df.as_matrix(columns=df.columns[:1])
y = y.reshape(y.shape[0])

my_clf = MyDecisionTreeClassifier(min_samples_split=2)
clf = DecisionTreeClassifier(min_samples_split=2)

t1 = time()
my_clf.fit(x, y)
t2 = time()
print(t2 - t1)

t1 = time()
clf.fit(x, y)
t2 = time()
print(t2 - t1)

print "#########"

gkf = KFold(n_splits=5, shuffle=True)

for train, test in gkf.split(x, y):
    X_train, y_train = x[train], y[train]
    X_test, y_test = x[test], y[test]
    my_clf.fit(X_train, y_train)
    print(accuracy_score(y_pred=my_clf.predict(X_test), y_true=y_test))

print "########"

for train, test in gkf.split(x, y):
    X_train, y_train = x[train], y[train]
    X_test, y_test = x[test], y[test]
    clf.fit(X_train, y_train)
    print(accuracy_score(y_pred=clf.predict(X_test), y_true=y_test))