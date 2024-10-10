from classes.DataClassifier import DataClassifierClass
from classes.DataExplorer import DataExplorerClass
from classes.DataSplit import DataSplit
from classes.DataWrangler import DataWranglerClass
from classes.FairMetric import FairMetricClass
from utils.LogHandling import log_val
import pandas as pd
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the UndefinedMetricWarning
warnings.filterwarnings("ignore", category = UndefinedMetricWarning)


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    # # Dataset settings: https://archive.ics.uci.edu/dataset/320/student+performance
    # data_path = 'C:/Users/ragha/Desktop/Files and Folders/Books and Semester Studies [Drive]/Fairness Clustering/Fairness Code/Fair-data-splits/data/student-UCI.csv'
    # columns_to_hot_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
    # yn_tf_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    # target_column_name = 'G3'
    # sensitive_attribute_column = 'sex_is_F'
    # sensitive_attribute_value = 1
    # columns_to_remove_right_before_classification = None

    # # Parameter settings
    # minimum_correlation_threshold = 0.1
    # split_ratio = 0.2
    # use_model = 'SVC'
    # iterations = 1000
    # fair_metric_list = [
    #         'statistical_parity',
    #         # 'conditional_parity',
    #         'equalized_odds',
    #         'equal_opportunity',
    #         'predictive_equality',
    #         'conditional_use_accuracy_equality',
    #         'predictive_parity',
    #         'overall_accuracy_equality'
    # ]

    # # Dataset settings: https://archive.ics.uci.edu/dataset/2/adult  (note: contains missing values)
    # data_path = "/Users/pranavchatur/Downloads/adult - Copy.csv"
    # columns_to_hot_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    # yn_tf_columns = None
    # target_column_name = 'target'
    # sensitive_attribute_column = 'marital-status_is_Married-civ-spouse'
    # sensitive_attribute_value = 1
    # columns_to_remove_right_before_classification = None

    # # Parameter settings
    # minimum_correlation_threshold = 0.1
    # split_ratio = 0.15
    # use_model = 'DeT'
    # iterations = 100
    # fair_metric_list = [
    #     'statistical_parity',
    #     # 'conditional_parity',
    #     'equalized_odds', 'equal_opportunity', 'predictive_equality', 'conditional_use_accuracy_equality', 'predictive_parity', 'overall_accuracy_equality'
    # ]

    # Dataset settings: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
    data_path = "/Users/pranavchatur/Downloads/data.csv"
    columns_to_hot_encode = ['Application mode', 'Course', 'Marital status', 'Previous qualification', 'Nacionality', 'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation']
    yn_tf_columns = None
    target_column_name = 'Target'
    sensitive_attribute_column = 'Gender'
    sensitive_attribute_value = 0
    columns_to_remove_right_before_classification = None

    # Parameter settings
    minimum_correlation_threshold = 0.1
    split_ratio = 0.25
    use_model = 'SVC'
    iterations = 100
    fair_metric_list = [
        'statistical_parity',
        # 'conditional_parity',
        'equalized_odds', 'equal_opportunity', 'predictive_equality', 'conditional_use_accuracy_equality', 'predictive_parity', 'overall_accuracy_equality'
    ]

    # # Dataset settings: https://archive.ics.uci.edu/dataset/117/census+income+kdd (300000 instances, used for finding first-order statistics of balance_for_split
    # data_path = "/Users/pranavchatur/Downloads/census/census-income.csv"
    # columns_to_hot_encode = ['sex']
    # yn_tf_columns = None
    # target_column_name = 'Target'
    # sensitive_attribute_column = 'sex_is_Male'
    # sensitive_attribute_value = 1
    # columns_to_remove_right_before_classification = None

    # # Parameter settings
    # minimum_correlation_threshold = 0.1
    # split_ratio = 0.25
    # use_model = 'SVC'
    # iterations = 1000
    # fair_metric_list = [
    #     'statistical_parity',
    #     # 'conditional_parity',
    #     'equalized_odds', 'equal_opportunity', 'predictive_equality', 'conditional_use_accuracy_equality', 'predictive_parity', 'overall_accuracy_equality'
    # ]

    evaluation_dataFrame_array = []

    # Data Wrangling
    dataset = DataWranglerClass(data_path, cols_to_encode = columns_to_hot_encode)
    dataset.convert_yn_tf_to_binary(cols_to_encode = yn_tf_columns)
    df = dataset.get_processed_dataframe()

    # Data Exploring here
    # data_explorer = DataExplorerClass(df, target_column_name = target_column_name)
    # log_val(data_explorer.trim_columns_with_correlation_less_than(min_correlation = minimum_correlation_threshold), disable = True)

    for i in range(iterations):

        evaluation_dataFrame_row = {}

        # Data Splitting here
        dataSplitter = DataSplit(df, sensitive_attribute_column, sensitive_attribute_value)
        train_set, test_set, balance_for_split = dataSplitter.random_split(split_ratio)
        print("Balance for split -> ", balance_for_split)
        evaluation_dataFrame_row['balance_for_split'] = balance_for_split

        # Data Classifying here
        clf = DataClassifierClass(train_set, test_set, target_column_name = target_column_name, columns_to_remove_pre_processing = columns_to_remove_right_before_classification)
        clf.evaluate_using_model(classifer_name = use_model)
        log_val(clf.predicted_target_appended_test_set(), disable = True)
        log_val(clf.performance_metrics())
        log_val(clf.performance_through_confusion_matrix(), disable = True)
        evaluation_dataFrame_row.update(clf.performance_metrics())

        # Fair Metric calculator here
        fair_metric = FairMetricClass(clf.predicted_target_appended_test_set(), sensitive_attribute_column, sensitive_attribute_value, target_column_name, target_column_name + '_predicted')
        fair_metric_results = fair_metric.check_multiple(fair_metric_list)
        print(fair_metric_results)
        evaluation_dataFrame_row.update(fair_metric_results)

        evaluation_dataFrame_array.append(evaluation_dataFrame_row)

    evaluation_dataFrame = pd.DataFrame(evaluation_dataFrame_array)
    final_data_explorer = DataExplorerClass(evaluation_dataFrame, target_column_name = 'balance_for_split')
    # log_val(final_data_explorer.correlation_values_dataframe())
    log_val(final_data_explorer.trim_columns_with_correlation_less_than(min_correlation = minimum_correlation_threshold).columns)
    return 1


if __name__ == "__main__":
    all_functions('run.py')
