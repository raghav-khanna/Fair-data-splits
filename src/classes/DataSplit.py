import random
import pandas as pd
import time


class DataSplit:
    # check inbuilt sklearn datasplit function as well

    def __init__(self, dataset: pd.DataFrame) -> None:
        # self.train_set = dataset.loc[0:19]
        # self.test_set = dataset.loc[20:26]
        temp = pd.DataFrame({1:[1,2,1,2,3,4,5],2:[1,3,1,2,3,4,5],3:[1,4,1,2,3,4,5],4:[1,5,1,2,3,4,5],5:[1,6,1,2,3,4,5]})
        time.sleep(2)
        self.train_set = dataset.sample(frac=0.85, random_state=int(time.time()))
        self.test_set = dataset.drop(self.train_set.index)

    def __randomlySplit(dataset: [], trainTestRatio: float = 0.1) -> tuple:
        testSetSize: int = len(dataset) * trainTestRatio
        testSet: [] = random.sample(dataset, testSetSize)
        trainSet: [] = dataset - testSet
        return trainSet, testSet
