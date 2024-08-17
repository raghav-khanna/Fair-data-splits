'''

TODO:

Data Explorer:
1. Correlation values
2. Draw visualizations
3. Principle Component Analysis (PCA)


'''
import pandas as pd
import numpy as np
from utils.LogHandling import log_err, log_prog
np.seterr(invalid='ignore')


class DataExplorerClass:

    __acceptable_column_datatypes = ['int64']

    def __init__(self, dataset: pd.DataFrame, target_column_name: str = '') -> None:
        log_prog('Enter classes/' + type(self).__name__ + '.constructor')
        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if dataset.empty:
            log_err('Dataset provided is empty, please provide at least 2 columns in the dataset')
            premature_return = True
        if target_column_name == '':
            log_err('Please provide the column name which contains the target attribute (attribute which needs to be predicted)')
            premature_return = True
        if target_column_name not in dataset:
            log_err('Specified target column "' + target_column_name + '" does not exist in provided train/test dataset, please modify accordingly')
            premature_return = True
        if premature_return:
            return
        log_prog('Complete parameter pre-checks')
        for column in dataset:
            if column != target_column_name and dataset[column].dtype in self.__acceptable_column_datatypes:
                try:
                    cor = np.corrcoef(dataset[column], dataset[target_column_name])
                    # print('Pearson Correlation value between column "' + str(column) + '" and "' + str(target_column_name) + '" is ' + str(cor[0][1]))
                except Exception:
                    log_err('Column "' + target_column_name + '" does not change, hence resulting in division by zero error')
                    raise
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')
