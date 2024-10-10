import random
import pandas as pd
import time
import logging

logging.getLogger('requests').setLevel(logging.ERROR)
logging.basicConfig(level = 30, format = "%(levelname)s:%(message)s:\n")  # Comment this line to stop showing the messages


class DataSplit:
    # check inbuilt sklearn datasplit function as well

    def __init__(self, dataset: pd.DataFrame, sensitive_attribute_column: str, sensitive_attribute_value: int) -> None:
        logging.info('********** Enter DataSplit constructor **********')
        self.__dataset: pd.DataFrame = dataset
        self.__sensitive_attribute_column: str = sensitive_attribute_column
        self.__sensitive_attribute_value: int = sensitive_attribute_value

    def random_split(self, split_ratio: float = 0.1, random_state: int = 1) -> tuple:
        logging.info("*****Random Splitting*****")
        # time.sleep(0.5)
        dataset = self.__dataset
        test_set = dataset.sample(frac = split_ratio)
        train_set = dataset.drop(test_set.index)
        balance_for_split = self.__calculate_balance_for_split((train_set, test_set))
        return train_set, test_set, balance_for_split

    def __calculate_balance_for_split(self, split: tuple) -> float:
        logging.info("*****Calculating balance for split*****")
        train_set, test_set = split
        dataset_ratio: float = len(self.__dataset[self.__dataset[self.__sensitive_attribute_column] == self.__sensitive_attribute_value]) / len(self.__dataset)
        train_set_ratio: float = len(train_set[train_set[self.__sensitive_attribute_column] == self.__sensitive_attribute_value]) / len(train_set)
        test_set_ratio: float = len(test_set[test_set[self.__sensitive_attribute_column] == self.__sensitive_attribute_value]) / len(test_set)
        if dataset_ratio == 0:
            logging.error("Dataset ratio is zero")
            return 0
        elif train_set_ratio == 0:
            logging.debug("Train set ratio is zero")
            train_set_balance: float = 0
            return 0
        elif test_set_ratio == 0:
            logging.debug("Test set ratio is zero")
            test_set_balance: float = 0
            return 0
        test_set_balance: float = min(test_set_ratio / dataset_ratio, dataset_ratio / test_set_ratio)
        train_set_balance: float = min(train_set_ratio / dataset_ratio, dataset_ratio / train_set_ratio)
        logging.info(f"Train set balance: {train_set_balance}")
        logging.info(f"Test set balance: {test_set_balance}")
        return min(train_set_balance, test_set_balance)
