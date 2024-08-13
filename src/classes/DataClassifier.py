'''

TODO: Document the file-functioning well.

DataClassifier:
- Methods
    - evaluate_using

'''
from types import FunctionType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from typing import Dict as dict
from typing import List as list
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from utils.LogHandling import log_err, log_prog
from typing import Union


class DataClassifierClass:
    __classifier_mapper: dict[str, FunctionType] = {'KNN': KNeighborsClassifier, 'SVC': SVC, 'GNB': GaussianNB, 'DeT': DecisionTreeClassifier}

    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target_column_name = '', classifer_name: str = 'KNN', columns_to_remove_preprocessing: Union[list[str], None] = []) -> None:
        log_prog('Enter classes/' + type(self).__name__ + '.constructor')

        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if target_column_name == '':
            log_err('Please provide the column name which contains the target attribute (attribute which needs to be predicted)')
            premature_return = True
        if target_column_name not in train_set or target_column_name not in test_set:
            log_err('Specified target column "' + target_column_name + '" does not exist in provided train/test dataset, please modify accordingly')
            premature_return = True
        if str(train_set.columns) != str(test_set.columns):
            log_err('The columns for train set( ' + str(train_set.columns) + ') and test set( ' + str(test_set.columns) + ') do not match. Please ensure that both have identical columns')
            premature_return = True
        if columns_to_remove_preprocessing is not None and len(columns_to_remove_preprocessing) == 0:
            log_err('Please pass the column names (primary keys) which are to be removed before processing, else pass "None" if none exist')
            premature_return = True
        if premature_return:
            self.__x_train: [] = []
            self.__x_test: [] = []
            self.__y_train: [] = []
            self.__y_test: [] = []
            self.__y_predict: [] = []
            self.__result: [] = []
            return
        log_prog('Complete parameter pre-checks')
        log_prog('Initialise training and testing dataset')
        copy_of_train_set = train_set.copy()
        copy_of_test_set = test_set.copy()
        if columns_to_remove_preprocessing is not None:
            log_prog('Remove ' + str(columns_to_remove_preprocessing) + ' columns from a copy of the provided dataset')
            for column in columns_to_remove_preprocessing:
                del train_set[column]
                del test_set[column]
        else:
            log_prog('No columns to remove pre-processing')
        log_prog('Segregate labels/features/attributes from target label/feature/attribute')
        self.__y_train: [] = copy_of_train_set.pop(target_column_name).values.tolist()
        self.__y_test: [] = copy_of_test_set.pop(target_column_name).values.tolist()
        self.__x_train: [] = copy_of_train_set.values.tolist()
        self.__x_test: [] = copy_of_test_set.values.tolist()
        self.__y_predict: [] = []
        self.__result: [] = []
        print('Model to be trained on: ', self.__x_train, 'with targets: ', self.__y_train)
        print('Model to be tested on: ', self.__x_test, 'with targets: ', self.__y_test)
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')

    def evaluate_using(self, classifer_name: str = '', **kwargs):
        log_prog('Enter classes/' + type(self).__name__ + '.evaluate')
        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if classifer_name not in self.__classifier_mapper.keys():
            log_err('Invalid or Unsupported Classifier name, only following supported: ' + str(self.__classifier_mapper.keys())[11:-2])
            premature_return = True
        if (len(self.__x_train) and len(self.__x_test) and len(self.__y_train) and len(self.__y_test)) == 0:
            log_prog('Object configured incorrectly, it cannot be trained. Please pass correct data in the class constructor')
            premature_return = True
        if premature_return:
            return
        log_prog('Complete parameter pre-checks')
        parameterised_classifier = []
        if self.__classifier_mapper[classifer_name] == KNeighborsClassifier:
            k = int(kwargs.get('k'))
            parameterised_classifier = Pipeline(steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors = k))])
        elif self.__classifier_mapper[classifer_name] == SVC:
            # TODO: Check relevant config options
            parameterised_classifier = SVC()
        elif self.__classifier_mapper[classifer_name] == GaussianNB:
            # TODO: Check relevant config options
            parameterised_classifier = GaussianNB()
        elif self.__classifier_mapper[classifer_name] == DecisionTreeClassifier:
            # TODO: Check relevant config options
            parameterised_classifier = DecisionTreeClassifier()
        parameterised_classifier.fit(self.__x_train, self.__y_train)
        self.__y_predict: [float | str | int] = parameterised_classifier.predict(self.__x_test)
        self.__result: [[int, [float | int], str | float | int, str | float | int]] = []
        self.__confusionMatrix = confusion_matrix(self.__y_test, self.__y_predict)
        for i in range(len(self.__x_test)):
            self.__result.append([i + 1, self.__x_test[i], self.__y_test[i], self.__y_predict[i]])

        print('**********\nPerformance of', classifer_name, ': ')
        print('\nTrained on:')
        print('|Attributes|Target|')
        for i in range(len(self.__x_train)):
            print('|', self.__x_train[i], '  |  ', self.__y_train[i], '  |')
        print('\n\nTest Results:')
        print('|Id|Attributes|Expected|Predicted|')
        for i in range(len(self.__x_test)):
            print('|', self.__result[i][0], '|', self.__result[i][1], '  |  ', self.__result[i][2], '  |  ', self.__result[i][3], '|')
        print('\n\n**********')
        print('\n\nConfusion Matrix:')
        print('| True Negative:', self.__confusionMatrix[0][0], '|False Positive:', self.__confusionMatrix[0][1], '|')
        print('|False Negative:', self.__confusionMatrix[1][0], '| True Positive:', self.__confusionMatrix[1][1], '|')
        print('\n\n**********')
        log_prog('Exit classes/' + type(self).__name__ + '.evaluate')
        return self.__result
