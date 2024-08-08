from types import FunctionType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
'''

TODO: Document the file-functioning well.

Classifier:
- Implements
    - Data Cleaning
    - Data One-Shot Encoding
- Methods
    - ...

'''
from sklearn.preprocessing import StandardScaler
from utils.LogHandling import log_prog


class DataClassifierClass:
    __classifier_mapper: dict[str, FunctionType] = {'KNN': KNeighborsClassifier, 'SVC': SVC, 'GNB': GaussianNB, 'DeT': DecisionTreeClassifier}

    def __init__(self, train_set: [], test_set: [], classifer_name: str = 'KNN', ) -> None:
        log_prog('Enter classes/' + type(self).__name__ + '.constructor')
        self.__classifer_name: str = classifer_name
        self.__x_train: [] = [vals[0:len(vals) - 1] for vals in train_set]
        self.__x_test: [] = [vals[0:len(vals) - 1] for vals in test_set]
        self.__y_train: [] = [vals[-1] for vals in train_set]
        self.__y_test: [] = [vals[-1] for vals in test_set]
        self.__y_predict: [] = []
        self.__result: [] = []
        print('To be trained on: ', self.__x_train, 'with targets: ', self.__y_train)
        print('To be tested on: ', self.__x_test, 'with targets: ', self.__y_test)
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')

    def evaluate(self, **kwargs):
        log_prog('Enter classes/' + type(self).__name__ + '.evaluate')
        parameterised_classifier = []
        if self.__classifier_mapper[self.__classifer_name] == KNeighborsClassifier:
            k = int(kwargs.get('k'))
            parameterised_classifier = Pipeline(steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors = k))])
        elif self.__classifier_mapper[self.__classifer_name] == SVC:
            # To check for relevant config options
            parameterised_classifier = SVC()
        elif self.__classifier_mapper[self.__classifer_name] == GaussianNB:
            # To check for relevant config options
            parameterised_classifier = GaussianNB()
        elif self.__classifier_mapper[self.__classifer_name] == DecisionTreeClassifier:
            # To check for relevant config options
            parameterised_classifier = DecisionTreeClassifier()
        parameterised_classifier.fit(self.__x_train, self.__y_train)
        self.__y_predict: [float | str | int] = parameterised_classifier.predict(self.__x_test)
        self.__result: [[int, [float | int], str | float | int, str | float | int]] = []
        for i in range(len(self.__x_test)):
            self.__result.append([i + 1, self.__x_test[i], self.__y_test[i], self.__y_predict[i]])
        log_prog('Exit classes/' + type(self).__name__ + '.evaluate')
        return self.__result

    def performance(self):
        print('**********\nPerformance of', self.__classifer_name, ': ')
        print('\nTrained on:')
        print('|Attributes|Target|')
        for i in range(len(self.__x_train)):
            print('|', self.__x_train[i], '  |  ', self.__y_train[i], '  |')
        print('\n\nTest Results:')
        print('|Id|Attributes|Expected|Predicted|')
        for i in range(len(self.__x_test)):
            print('|', self.__result[i][0], '|', self.__result[i][1], '  |  ', self.__result[i][2], '  |  ', self.__result[i][3], '|')
        print('\n\n**********')
