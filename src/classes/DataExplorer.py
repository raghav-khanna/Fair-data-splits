'''

Class Data Explorer:
- Implements
    - Indicate presence of cardinal attributes making dataset unfit for training a model
    - Generate dataframe of correlation values between features and target attribute
    - Trim provided dataframe based on provided minimum correlation value between features and target attribute
    - Return all the unique target labels present in the dataset

- Parameters in the constructor
    - dataframe: pd.Dataframe => DataFrame that contains the data to be explored. Warning is thrown if the data contains cardinal features.
    - target_column_name: str => Name of the target attribute

- Public methods
    - unique_target_column_values: Returns a list of all the unique target labels present in the dataset
    - correlation_values_dataframe: Returns a dataframe whose headers are the dataset attributes and the single row contains correlation values between target attribute and corresponding feature
    - trim_columns_with_correlation_less_than: Return a modified dataframe which contains only those features which have correlation value greater than the user-specified value
        - min_correlation: float = Takes input the minimum correlation value for the feature to be used for training the model. Must be between 0 and 1

- TODO
    - Plot visualizations to understand how the data looks like
    - Principle Component Analysis (PCA)

'''

import pandas as pd
import numpy as np
from typing import Dict as dict
from utils.LogHandling import log_err, log_prog, log_val

np.seterr(invalid = 'ignore')


class DataExplorerClass:

    __acceptable_column_datatypes = ['int8', 'int32', 'int64', 'uint8', 'uint32', 'uint64', 'float32', 'float64']

    def __init__(self, dataframe: pd.DataFrame = pd.DataFrame(), target_column_name: str = '') -> None:
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
        cardinal_dataframe: pd.DataFrame = self.__dataframe.select_dtypes(exclude = self.__acceptable_column_datatypes)
        for column in cardinal_dataframe.columns:
            if column != self.__target_column_name:
                log_err('Column "' + str(column) + '" is not of type numeric. The dataset is unfit for training the model. Please remove or one-hot encode the column')
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

    def trim_columns_with_correlation_less_than(self, min_correlation: float = 0.0) -> pd.DataFrame:
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
