import random


def randomlySplit(dataset: [], trainTestRatio: float = 0.1) -> tuple:
    testSetSize: int = len(dataset) * trainTestRatio
    testSet: [] = random.sample(dataset, testSetSize)
    trainSet: [] = dataset - testSet
    return trainSet, testSet


class DataSplit:
    #check inbuilt sklearn datasplit function as well

    def __init__(self, dataset: []) -> None:
        pass
