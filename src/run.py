from classes.DataClassifier import DataClassifierClass
from classes.DataExplorer import DataExplorerClass
from classes.DataSplit import DataSplit
from classes.DataWrangler import DataWranglerClass
# import numpy as np


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    dataset = DataWranglerClass('<abs-path>/Fair-data-splits/data/student.csv', cols_to_encode = None)
    # dataset.convert_yn_tf_to_binary(cols_to_encode=['age'])
    dataset.convert_yn_tf_to_binary(cols_to_encode = None)
    df = dataset.get_processed_dataframe()
    print(df)

    DataExplorerClass(df, target_column_name = 'hired')

    # Data Splitting here
    dataSplitter = DataSplit(df)
    train_set = dataSplitter.train_set
    test_set = dataSplitter.test_set

    # Data Exploring here
    # ls1 = df['studytime'].values.tolist();ls2 = df['hired'].values.tolist()
    # print(ls1);print(ls2)
    # if len(ls1) == len(ls2):
    #     cor = np.corrcoef(ls1, ls2)
    #     print(cor)

    # Data Classifying here
    clf = DataClassifierClass(train_set, test_set, target_column_name = 'hired', columns_to_remove_preprocessing = ['name'])
    results = clf.evaluate_using(k = 2, classifer_name = 'KNN')
    print(results)

    # Classifier Performance here
    return 1


if __name__ == "__main__":
    all_functions('run.py')
