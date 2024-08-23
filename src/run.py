from classes.DataClassifier import DataClassifierClass
from classes.DataExplorer import DataExplorerClass
from classes.DataSplit import DataSplit
from classes.DataWrangler import DataWranglerClass
from utils.LogHandling import log_val


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    # Variable setting
    data_path = ''
    columns_to_hot_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']
    yn_tf_columns = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    target_column_name = 'G3'
    minimum_correlation_threshold = 0.1
    columns_to_remove_right_before_classification = None
    use_model = 'SVC'

    # Data Wrangling
    dataset = DataWranglerClass(data_path, cols_to_encode = columns_to_hot_encode)
    dataset.convert_yn_tf_to_binary(cols_to_encode = yn_tf_columns)
    df = dataset.get_processed_dataframe()

    # Data Exploring here
    data_explorer = DataExplorerClass(df, target_column_name = target_column_name)
    log_val(data_explorer.trim_columns_with_correlation_less_than(min_correlation = minimum_correlation_threshold))

    # Data Splitting here
    dataSplitter = DataSplit(df)
    train_set = dataSplitter.train_set
    test_set = dataSplitter.test_set

    # Data Classifying here
    clf = DataClassifierClass(train_set, test_set, target_column_name = target_column_name, columns_to_remove_pre_processing = columns_to_remove_right_before_classification)
    results = clf.evaluate_using_model(classifer_name = use_model)
    log_val(results)

    # Fair Metric calculator here

    return 1


if __name__ == "__main__":
    all_functions('run.py')
