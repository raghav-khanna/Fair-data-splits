from classes.DataClassifier import DataClassifierClass
from classes.DataExplorer import DataExplorerClass
from classes.DataSplit import DataSplit
from classes.DataWrangler import DataWranglerClass
from classes.FairMetric import FairMetricClass

# import numpy as np


def all_functions(name: str) -> int:
    path = 'C:/Users/ragha/Desktop/Files and Folders/Books and Semester Studies [Drive]/Fairness Clustering/Fairness Code/Fair-data-splits/data/student-UCI.csv'
    dataset = DataWranglerClass(path, cols_to_encode = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'])
    # dataset.convert_yn_tf_to_binary(cols_to_encode=['age'])
    dataset.convert_yn_tf_to_binary(cols_to_encode = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])
    df = dataset.get_processed_dataframe()
    # print(df)

    DataExplorerClass(df, target_column_name = 'G3')

    # Data Splitting here
    for i in range(0,100):
        dataSplitter = DataSplit(df)
        train_set = dataSplitter.train_set
        test_set = dataSplitter.test_set
        # print("**************************************")
        # print(train_set)
        # print("**************************************")
        # print(test_set)

        # Data Classifying here
        clf = DataClassifierClass(train_set, test_set, target_column_name = 'G3', columns_to_remove_preprocessing = None)
        results = clf.evaluate_using(classifer_name = 'GNB')
    # print(results)

    # fairMetricObj = FairMetricClass(results, 3, 1)
    # fairMetricObj.check_multiple(['statistical_parity','equalized_odds','equal_opportunity','predictive_equality','conditional_use_accuracy_equality','predictive_parity','overall_accuracy_equality'])

    # Classifier Performance here
    return 1


if __name__ == "__main__":
    all_functions('run.py')
