'''

Class Pipeline Runner
- Implements
    - Run Classification Pipeline (Wrangling, Exploring, Spltting, Classifying)

- Parameters in the constructor
    - file_path: str => Absolute path of the file containing the data (currently only .csv supported)
    - target_column_name: str => Name of the target column in the dataset
    - classifer_name: str => Name of the classifier to be used for classification
    - cols_to_hot_encode: list[str] => Names of the attributes which are in string and need to be converted into 0/1, i.e. one hot encoded (pass None if no such column)
    - yn_tf_cols: list[str] => Names of the columns which contains values in the form of yes/no/true/false and need to be converted to binary (pass None if no such column)
    - minimum_correlation_threshold: float = Takes input the minimum correlation value for the feature to be used for training the model. Must be between 0 and 1
    - columns_to_remove_pre_processing: list[str] => List of column names to remove before training the model (like id, name...). Pass None if no such column
    - sensitive_attribute_column: str => Name of the sensitive attribute column
    - sensitive_attribute_value: int => Value encoded for the protected group in the sensitive_attribute_column
    - split_ratio: float => The split ratio of train and test. 0.2 implies 20% test and 80% train
    - iterations: int => Number of times a new data split has to be generated and model to be trained and evaluated over the new split
    - fair_metrics_list: list[str] => List of fair metrics to calculate post model evaluation

- Public Methods
    - run_classification_pipeline
        - kwargs
            - k: int => K of the KNeighborsClassifier in case of KNN classifier

- TODO
    - Setup DataClusterer Class
    - Setup Reinforcement Learner
    - Setup whatever LLMs are

'''

from typing import Union
from classes.DataClassifier import DataClassifierClass
from classes.DataExplorer import DataExplorerClass
from classes.DataSplit import DataSplit
from classes.DataWrangler import DataWranglerClass
from classes.FairMetric import FairMetricClass
from utils.LogHandling import log_val
import pandas as pd


class PipelineRunnerClass:

    def __init__(
        self,
        file_path: str,
        target_column_name: str,
        classifier_name: str,
        sensitive_attribute_column: str,
        sensitive_attribute_value: int,
        split_ratio: float,
        iterations: int,
        fair_metrics_list: list[str],
        columns_to_hot_encode: Union[list[str], None] = None,
        yn_tf_columns: Union[list[str], None] = None,
        minimum_correlation_threshold: float = 0,
        columns_to_remove_right_before_classification: Union[list[str], None] = None
    ) -> None:
        # Dataset settings
        self.__file_path = file_path
        self.__columns_to_hot_encode = columns_to_hot_encode
        self.__yn_tf_columns = yn_tf_columns
        self.__target_column_name = target_column_name
        self.__columns_to_remove_right_before_classification = columns_to_remove_right_before_classification
        self.__sensitive_attribute_column = sensitive_attribute_column
        self.__sensitive_attribute_value = sensitive_attribute_value

        # Parameter settings
        self.__minimum_correlation_threshold = minimum_correlation_threshold
        self.__split_ratio = split_ratio
        self.__classifier_name = classifier_name
        self.__iterations = iterations
        self.__fair_metric_list = fair_metrics_list

    def run_classification_pipeline(self, **kwargs):

        sep = kwargs.get('sep', ';')

        evaluation_dataFrame_array = []

        # Data Wrangling
        dataset = DataWranglerClass(self.__file_path, cols_to_hot_encode = self.__columns_to_hot_encode, yn_tf_cols = self.__yn_tf_columns, sep = sep)
        df = dataset.get_processed_dataframe()

        # Data Exploring
        data_explorer = DataExplorerClass(df, target_column_name = self.__target_column_name)
        log_val(data_explorer.correlation_values_dataframe())
        log_val(data_explorer.trim_columns_with_correlation_less_than(min_correlation = self.__minimum_correlation_threshold))

        for _ in range(self.__iterations):

            evaluation_dataFrame_row = {}

            # Data Splitting here
            dataSplitter = DataSplit(df, self.__sensitive_attribute_column, self.__sensitive_attribute_value)
            train_set, test_set, balance_for_split = dataSplitter.random_split(self.__split_ratio)
            log_val("Balance for split -> " + str(balance_for_split))
            evaluation_dataFrame_row['balance_for_split'] = balance_for_split

            # Data Classifying
            clf = DataClassifierClass(train_set, test_set, target_column_name = self.__target_column_name, columns_to_remove_pre_processing = self.__columns_to_remove_right_before_classification)
            clf.evaluate_using_model(classifer_name = self.__classifier_name)
            log_val(clf.predicted_target_appended_test_set())
            log_val(clf.performance_metrics())
            log_val(clf.performance_through_confusion_matrix())
            evaluation_dataFrame_row.update(clf.performance_metrics())

            # Fair Metric calculator
            fair_metric = FairMetricClass(clf.predicted_target_appended_test_set(), self.__sensitive_attribute_column, self.__sensitive_attribute_value, self.__target_column_name, self.__target_column_name + '_predicted')
            fair_metric_results = fair_metric.check_multiple(self.__fair_metric_list)
            print(fair_metric_results)
            evaluation_dataFrame_row.update(fair_metric_results)

            evaluation_dataFrame_array.append(evaluation_dataFrame_row)

        evaluation_dataFrame = pd.DataFrame(evaluation_dataFrame_array)
        final_data_explorer = DataExplorerClass(evaluation_dataFrame, target_column_name = 'balance_for_split')
        # log_val(final_data_explorer.correlation_values_dataframe())
        log_val(final_data_explorer.trim_columns_with_correlation_less_than(min_correlation = self.__minimum_correlation_threshold).columns)
