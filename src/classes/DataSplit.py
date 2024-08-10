import random
import pandas as pd

class DataSplit:
    # check inbuilt sklearn datasplit function as well

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.train_set = dataset.loc[0:19]
        self.test_set = dataset.loc[20:26]
        

    def __randomlySplit(dataset: [], trainTestRatio: float = 0.1) -> tuple:
        testSetSize: int = len(dataset) * trainTestRatio
        testSet: [] = random.sample(dataset, testSetSize)
        trainSet: [] = dataset - testSet
        return trainSet, testSet
