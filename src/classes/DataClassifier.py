'''

TODO: Document the file-functioning well.

DataClassifier:
- Methods
    - evaluate_using_model
    - predicted_target_appended_test_set
    - performance_metrics
    - performance_through_confusion_matrix

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
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class DataClassifierClass:
    __classifier_mapper: dict[str, FunctionType] = {'KNN': KNeighborsClassifier, 'SVC': SVC, 'GNB': GaussianNB, 'DeT': DecisionTreeClassifier}
    __acceptable_column_datatypes = ['int8', 'int32', 'int64', 'uint8', 'uint32', 'uint64', 'float32', 'float64', 'boolean']
    __train_set: pd.DataFrame = pd.DataFrame()
    __test_set: pd.DataFrame = pd.DataFrame()
    __target_column_name: str = ''
    __performace_metrics: dict[str, float] = {}
    __updated_test_set: pd.DataFrame = pd.DataFrame()
    __confusion_matrix: list = []

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
            return
        log_prog('Complete parameter pre-checks')
        log_prog('Initialise training and testing dataset')
        self.__train_set: pd.DataFrame = train_set.copy()
        self.__test_set: pd.DataFrame = test_set.copy()
        self.__target_column_name: str = target_column_name
        if columns_to_remove_pre_processing is not None:
            log_prog('Remove ' + str(columns_to_remove_pre_processing) + ' columns from a copy of the provided dataset')
            for column in columns_to_remove_pre_processing:
                del self.__train_set[column]
                del self.__test_set[column]
        else:
            log_prog('No columns to remove pre-processing')
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')

    def evaluate_using_model(self, classifer_name: str = '', **kwargs):
        log_prog('Enter classes/' + type(self).__name__ + '.evaluate_using_model')
        copy_of_train_set = self.__train_set.copy()
        copy_of_test_set = self.__test_set.copy()
        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if classifer_name not in self.__classifier_mapper.keys():
            log_err('Invalid or Unsupported Classifier name, only following supported: ' + str(self.__classifier_mapper.keys())[11:-2])
            premature_return = True
        if self.__train_set.empty or self.__test_set.empty:
            log_prog('Object configured incorrectly, it cannot be trained. Please pass correct data in the class constructor')
            premature_return = True
        for column in copy_of_train_set:
            if column is not self.__target_column_name and copy_of_train_set[column].dtype not in self.__acceptable_column_datatypes:
                log_err('Column "' + str(column) + '" is not a number and is therefore dropped. Please one-hot-encode it using DataWrangler if it needs to be used for training')
                del copy_of_train_set[column]
                del copy_of_test_set[column]
        if premature_return:
            return
        log_prog('Complete parameter pre-checks')

        log_prog('Segregate labels/features/attributes from target label/feature/attribute')
        y_train: list = copy_of_train_set.pop(self.__target_column_name).values.tolist()
        y_test: list = copy_of_test_set.pop(self.__target_column_name).values.tolist()
        x_train: list = copy_of_train_set.values.tolist()
        x_test: list = copy_of_test_set.values.tolist()

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
        log_val('|Attributes|Target|',disable=True)
        # for i in range(len(x_train)):
            # log_val('|', x_train[i], '  |  ', y_train[i], '  |')
        parameterised_classifier.fit(x_train, y_train)

        log_prog('Test the model over testing data')
        y_predict: list[float | str | int] = parameterised_classifier.predict(x_test)

        log_prog('Append the predicted values of target column "' + str(self.__target_column_name) + '" as "' + str(self.__target_column_name) + '_predicted" to the dataframe')
        self.__updated_test_set = copy_of_test_set
        self.__updated_test_set[str(self.__target_column_name)] = y_test
        self.__updated_test_set[str(self.__target_column_name) + '_predicted'] = y_predict

        log_prog('Calculate metrics: accuract, precision, recall')
        self.__performace_metrics: dict[str, float] = {}
        self.__performace_metrics['accuracy'] = accuracy_score(y_test, y_predict)
        self.__performace_metrics['precision'] = precision_score(y_test, y_predict, average = 'macro')
        self.__performace_metrics['recall'] = recall_score(y_test, y_predict, average = 'macro')

        if len(set(y_test)) > 2:
            log_prog('Generate confusion matrices for multi-class classification')
            self.__confusion_matrix = multilabel_confusion_matrix(y_test, y_predict)
            # TODO: Print multiple confusion matrices gracefully
        else:
            log_prog('Generate confusion matrix for binary classification')
            self.__confusion_matrix = confusion_matrix(y_test, y_predict)
            log_val('| True Negative:', self.__confusion_matrix[0][0], '|False Positive:', self.__confusion_matrix[0][1], '|')
            log_val('|False Negative:', self.__confusion_matrix[1][0], '| True Positive:', self.__confusion_matrix[1][1], '|')
        log_prog('Exit classes/' + type(self).__name__ + '.evaluate_using_model')

    def performance_metrics(self) -> dict[str, float]:
        if not bool(self.__performace_metrics):
            log_err('Model evaluation is pending, performace metrics are undefined')
        return self.__performace_metrics

    def predicted_target_appended_test_set(self) -> pd.DataFrame:
        if self.__updated_test_set.empty:
            log_err('Model evaluation is pending, updated test set is empty')
        return self.__updated_test_set

    def performance_through_confusion_matrix(self):
        if len(self.__confusion_matrix) == 0:
            log_err('Model evaluation is pending, confusion matrix is empty')
        return self.__confusion_matrix
