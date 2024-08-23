'''

TODO: Document the file-functioning well.

DataClassifier:
- Methods
    - evaluate_using_model

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from utils.LogHandling import log_err, log_prog, log_val
from typing import Union


class DataClassifierClass:
    __classifier_mapper: dict[str, FunctionType] = {'KNN': KNeighborsClassifier, 'SVC': SVC, 'GNB': GaussianNB, 'DeT': DecisionTreeClassifier}
    __acceptable_column_datatypes = ['int8', 'int32', 'int64', 'uint8', 'uint32', 'uint64', 'float32', 'float64', 'boolean']

    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target_column_name = '', columns_to_remove_pre_processing: Union[list[str], None] = []) -> None:
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
        if columns_to_remove_pre_processing is not None and len(columns_to_remove_pre_processing) == 0:
            log_err('Please pass the column names (primary keys) which are to be removed before processing, else pass "None" if none exist')
            premature_return = True
        if premature_return:
            self.__x_train: list = []
            self.__x_test: list = []
            self.__y_train: list = []
            self.__y_test: list = []
            return
        log_prog('Complete parameter pre-checks')
        log_prog('Initialise training and testing dataset')
        copy_of_train_set = train_set.copy()
        copy_of_test_set = test_set.copy()
        if columns_to_remove_pre_processing is not None:
            log_prog('Remove ' + str(columns_to_remove_pre_processing) + ' columns from a copy of the provided dataset')
            for column in columns_to_remove_pre_processing:
                del copy_of_train_set[column]
                del copy_of_test_set[column]
        else:
            log_prog('No columns to remove pre-processing')

        for column in copy_of_train_set:
            if column is not target_column_name and copy_of_train_set[column].dtype not in self.__acceptable_column_datatypes:
                log_err('Column "' + str(column) + '" is not a number and is therefore dropped. Please one-hot-encode it using DataWrangler if it needs to be used for training')
                del copy_of_train_set[column]
                del copy_of_test_set[column]

        log_prog('Segregate labels/features/attributes from target label/feature/attribute')
        self.__y_train: list = copy_of_train_set.pop(target_column_name).values.tolist()
        self.__y_test: list = copy_of_test_set.pop(target_column_name).values.tolist()
        self.__x_train: list = copy_of_train_set.values.tolist()
        self.__x_test: list = copy_of_test_set.values.tolist()
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')

    def evaluate_using_model(self, classifer_name: str = '', **kwargs) -> pd.DataFrame:
        log_prog('Enter classes/' + type(self).__name__ + '.evaluate_using_model')
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

        # TODO: Add logic of sklearn flow chart to choose the classifier here as a default case

        parameterised_classifier = []
        log_prog('Select ' + str(classifer_name) + ' as a classifier')
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

        log_prog('Train the model over training data')
        log_val('|Attributes|Target|')
        for i in range(len(self.__x_train)):
            log_val('|', self.__x_train[i], '  |  ', self.__y_train[i], '  |')
        parameterised_classifier.fit(self.__x_train, self.__y_train)

        log_prog('Test the model over testing data')
        self.__y_predict: list[float | str | int] = parameterised_classifier.predict(self.__x_test)
        self.__combined_xtest_ytest_ypredict: list[list[list[float | int | str] | float | int | str]] = []
        for i in range(len(self.__x_test)):
            self.__combined_xtest_ytest_ypredict.append([self.__x_test[i], self.__y_test[i], self.__y_predict[i]])
        log_val('|Attributes|Expected|Predicted|')
        for i in range(len(self.__x_test)):
            log_val('|', self.__combined_xtest_ytest_ypredict[i][0], '|', self.__combined_xtest_ytest_ypredict[i][1], '  |  ', self.__combined_xtest_ytest_ypredict[i][2], '  |')

        log_prog('Calculate metrics: accuract, precision, recall')
        self.__performace_metrics: dict[str, float] = {}
        self.__performace_metrics['accuracy'] = accuracy_score(self.__y_test, self.__y_predict)
        self.__performace_metrics['precision'] = precision_score(self.__y_test, self.__y_predict, average = 'macro')
        self.__performace_metrics['recall'] = recall_score(self.__y_test, self.__y_predict, average = 'macro')
        log_val(self.__performace_metrics)

        if len(set(self.__y_test)) > 2:
            log_prog('Generate confusion matrices for multi-class classification')
            self.__confusion_matrix = multilabel_confusion_matrix(self.__y_test, self.__y_predict)
            log_val(self.__confusion_matrix)
            # TODO: Print multiple confusion matrices gracefully
        else:
            log_prog('Generate confusion matrix for binary classification')
            self.__confusion_matrix = confusion_matrix(self.__y_test, self.__y_predict)
            log_val('| True Negative:', self.__confusion_matrix[0][0], '|False Positive:', self.__confusion_matrix[0][1], '|')
            log_val('|False Negative:', self.__confusion_matrix[1][0], '| True Positive:', self.__confusion_matrix[1][1], '|')
        log_prog('Exit classes/' + type(self).__name__ + '.evaluate_using_model')
        return pd.DataFrame(data = self.__combined_xtest_ytest_ypredict, columns = ['x test', 'y test', 'predicted y'])

    # TODO: Add method to append to performace metrics to a list
