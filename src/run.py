from classes.DataClassifier import DataClassifierClass
from classes.DataWrangler import DataWranglerClass
from classes.DataSplit import DataSplit
from classes.FairMetric import FairMetric
import numpy as np


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    dataset = DataWranglerClass('C:/Users/ragha/Desktop/Files and Folders/Books and Semester Studies [Drive]/Fairness Clustering/Fairness Code/Fair-data-splits/data/student.csv', cols_to_encode = ['gender'])
    # dataset.convert_yn_tf_to_binary(cols_to_encode=['age'])
    dataset.convert_yn_tf_to_binary(cols_to_encode = None)
    df = dataset.get_processed_dataframe()
    # print(df)
    del df['gender_is_Male']
    df['hire'] = df['hired']
    del df['age']
    del df['name']
    del df['hired']
    # print(df)

    # Data Exploring here

    # ls1 = df['studytime'].values.tolist();ls2 = df['hired'].values.tolist()
    # print(ls1);print(ls2)
    # if len(ls1) == len(ls2):
    #     cor = np.corrcoef(ls1, ls2)
    #     print(cor)

    # Data Splitting here
    dataSplitter = DataSplit(df)
    # print(dataSplitter.test_set)

    train_set = dataSplitter.train_set.values.tolist()
    test_set = dataSplitter.test_set.values.tolist()

    clf = DataClassifierClass(train_set, test_set, classifer_name = 'KNN')
    results = clf.evaluate(k = 2)
    print(results)
    clf.performance()
    results = [x[1:] for x in results]

    fm = FairMetric(results, 2, 1)
    
    fm.allFairMetrics()
    # print(fairMetricResults)



    # Classifier Performance here
    return 1


if __name__ == "__main__":
    all_functions('run.py')
