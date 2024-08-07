from classes.DataClassifier import DataClassifierClass
from classes.DataWrangler import DataWranglerClass
import numpy as np


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    dataset = DataWranglerClass('<absolute-path>/Fair-data-splits/data/student.csv', cols_to_encode = ['gender', 'age'])
    print(dataset.get_processed_dataframe())
    ls1 = dataset['name'].values.tolist()
    ls2 = dataset['hired'].values.tolist()
    print(dataset)
    print(ls1)
    print(ls2)
    cor = np.corrcoef(ls1, ls2)
    print(cor)

    # Data Splitting here

    train_set = [[1, 1, 1], [2, 1, 1], [1, 2, 1], [-1, -1, 0], [-1, -2, 0], [-2, -1, 0]]
    test_set = [[2, 2, 1], [-2, -2, 0], [0, 0, 0], [3, 1, 0], [3, 1, 0]]
    clf = DataClassifierClass(train_set, test_set)
    results = clf.evaluate(k = 2)
    print(results)
    clf.performance()

    # Classifier Performance here
    return 1


if __name__ == "__main__":
    all_functions('run.py')
