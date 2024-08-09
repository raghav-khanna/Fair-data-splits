from classes.DataClassifier import DataClassifierClass
from classes.DataWrangler import DataWranglerClass
import numpy as np


def all_functions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    dataset = DataWranglerClass('/Users/pranavchatur/Fair-data-splits/data/student.csv', cols_to_encode = None)
    # dataset.convert_yn_tf_to_binary(cols_to_encode=['age'])
    dataset.convert_yn_tf_to_binary(cols_to_encode=None)
    df = dataset.get_processed_dataframe()
    print(df)

    ls1 = df['name'].values.tolist()
    ls2 = df['hired'].values.tolist()
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
