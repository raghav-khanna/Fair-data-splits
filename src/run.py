from classes.Classifier import ClassifierClass


def Allfunctions(name: str) -> int:
    # print('Import all functions to this file and run only this file: ', end = '')
    # print(name)

    trainSet = [[1, 1, 1], [2, 1, 1], [1, 2, 1], [-1, -1, 0], [-1, -2, 0],
                [-2, -1, 0]]
    testSet = [[2, 2, 1], [-2, -2, 0], [0, 0, 0], [3, 1, 0], [3, 1, 0]]
    clf = ClassifierClass(trainSet, testSet)
    results = clf.evaluate(k = 2)
    print(results)
    clf.performance()

    return 1


if __name__ == "__main__":
    Allfunctions('run.py')
