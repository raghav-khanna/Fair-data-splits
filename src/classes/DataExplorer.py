'''

Data Explorer:
1. Return list of unique target labels
2. Generate dataframe of correlation values between permissible attributes and target attribute
3. Trim provided dataframe based on minimum correlation value between permissible attributes and target attribute

TODO:
- Draw visualizations
- Principle Component Analysis (PCA)


'''

import pandas as pd
import numpy as np
from typing import Dict as dict
from utils.LogHandling import log_err, log_prog, log_val

np.seterr(invalid = 'ignore')


class DataExplorerClass:

    __acceptable_column_datatypes = ['int8', 'int32', 'int64', 'uint8', 'uint32', 'uint64', 'float32', 'float64', 'boolean']

    def __init__(self, dataframe: pd.DataFrame, target_column_name: str = '') -> None:
        log_prog('Enter classes/' + type(self).__name__ + '.constructor')
        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if dataframe.empty:
            log_err('Dataset provided is empty, please provide at least 2 columns in the dataset')
            premature_return = True
        if target_column_name == '':
            log_err('Please provide the column name which contains the target attribute (attribute which needs to be predicted)')
            premature_return = True
        if target_column_name not in dataframe:
            log_err('Specified target column "' + target_column_name + '" does not exist in provided train/test dataset, please modify accordingly')
            premature_return = True
        if premature_return:
            return
        log_prog('Complete parameter pre-checks')
        log_prog('Instantiate Object')
        self.__dataframe: pd.DataFrame = dataframe
        self.__target_column_name: str = target_column_name
        self.__target_column_values: list[str] = sorted(dataframe[target_column_name].unique())
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')

    def unique_target_column_values(self) -> list[str]:
        return self.__target_column_values

    def correlation_values_dataframe(self) -> pd.DataFrame:
        log_prog('Enter classes/' + type(self).__name__ + '.correlation_values_dataframe')
        dataframe_dict: dict[str, float | str] = {'target_column_name': self.__target_column_name}
        for column in self.__dataframe:
            if column != self.__target_column_name and self.__dataframe[column].dtype in self.__acceptable_column_datatypes:
                try:
                    cor = np.corrcoef(self.__dataframe[column], self.__dataframe[self.__target_column_name])
                    log_val('Pearson Correlation value between column "' + str(column) + '" and "' + str(self.__target_column_name) + '" is ' + str(cor[0][1]))
                    dataframe_dict[column] = cor[0][1]
                except Exception:
                    log_err('Column "' + self.__target_column_name + '" does not change, hence resulting in division by zero error')
                    dataframe_dict[column] = 0
                    raise
        log_prog('Exit classes/' + type(self).__name__ + '.correlation_values_dataframe')
        return pd.DataFrame(data = [dataframe_dict])

    def trim_columns_with_correlation_less_than(self, min_correlation: float = 0.0):
        log_prog('Enter classes/' + type(self).__name__ + '.trim_columns_with_correlation_less_than')
        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if abs(min_correlation) > 1:
            premature_return = True
        if premature_return:
            return
        log_prog('Complete parameter pre-checks')
        correlation_dataframe = self.correlation_values_dataframe()
        trimmed_dataframe = self.__dataframe.copy()
        for column in correlation_dataframe:
            if column == 'target_column_name':
                continue
            if abs(correlation_dataframe[column][0]) < min_correlation:
                log_prog('Trim column: ' + str(column))
                trimmed_dataframe.drop(column, axis = 1, inplace = True)

        log_prog('Exit classes/' + type(self).__name__ + '.trim_columns_with_correlation_less_than')
        return trimmed_dataframe
