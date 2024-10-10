'''

Class Data Wrangler:
- Implements
    - Read data
    - Convert data into Pandas Dataframe
    - Clean data
    - One-Shot Encode data (to convert cardinal attributes to ordinal attributes)
    - Encode yes/no, true/false columns to Binary format

- Parameters in the constructor
    - file_path: str => Absolute path of the file containing the data (currently only .csv supported)
    - handle_missing: str => Option to handle missing values in the data (currently only 'leave' is supported')
    - cols_to_hot_encode: list[str] => Names of the attributes which are in string and need to be converted into 0/1, i.e. one hot encoded (pass None if no such column)
    - yn_tf_cols: list[str] => Names of the columns which contains values in the form of yes/no/true/false and need to be converted to binary (pass None if no such column)
    - kwargs
        - sep: char => Seperator character when the data file is a csv (default => ,)

- Public Methods
    - get_processed_dataframe: Returns the data in a Pandas Dataframe, cleaned, one-hot-encoded and binarised as per inputs in the constructor

- TODO
    - Read from other file formats (apart from .csv)
    - Handle missing data
    - Consider scenarios where header is present in a separate file from the data in .csv format

'''

import os
from types import FunctionType
from typing import Dict as dict
from typing import List as list
import pandas as pd
from utils.LogHandling import log_err, log_prog
from typing import Union


class DataWranglerClass:

    __readers: dict[str, FunctionType] = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xlx': pd.read_excel,
        '.json': pd.read_json,
        '.xml': pd.read_xml,
        '.parquet': pd.read_parquet,
        '.feather': pd.read_feather,
        '.orc': pd.read_orc,
        '.pkl': pd.read_pickle,
        '.html': pd.read_html,
        '.h5': pd.read_hdf,
        '.sav': pd.read_spss,
        '.dta': pd.read_stata
        # Explore pd.read_sql, pd.read_sql_query, pd.read_sql_table
    }

    __handle_missing_options: list[str] = ['delete', 'leave', 'error', 'mean', 'median']

    def __init__(self, file_path: str, handle_missing: str = 'leave', cols_to_hot_encode: Union[list[str], None] = [], yn_tf_cols: Union[list[str], None] = [], **kwargs):
        log_prog('Enter classes/' + type(self).__name__ + '.constructor')

        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if handle_missing not in self.__handle_missing_options:
            log_err('Invalid handle_missing option provided')
            premature_return = True
        if cols_to_hot_encode is not None and len(cols_to_hot_encode) == 0:
            log_err('Please provide columns which are to be one-hot encoded, or pass "None"')
            premature_return = True
        if yn_tf_cols is not None and len(yn_tf_cols) == 0:
            log_err('Please provide columns which are in yes/no or true/false, or pass "None"')
            premature_return = True
        if premature_return:
            self.__dataset = []
            return
        log_prog('Complete parameter pre-checks')

        log_prog('Retrieve data from provided location')
        file_extension: str = os.path.splitext(file_path)[1]
        if file_extension == '.csv':
            seperator = kwargs.get('sep', ',')
            self.__dataset: pd.DataFrame = self.__readers[file_extension](file_path, sep = seperator)
        else:
            log_err('Only .csv files are supported AS OF NOW')
            exit()

        log_prog('Clean data')
        self.__dataset = self.__clean_data(self.__dataset, handle_missing)

        if cols_to_hot_encode is not None:
            log_prog('One-hot encode data')
            self.__dataset = self.__one_hot_encoder(self.__dataset, cols_to_hot_encode)

        if yn_tf_cols is not None:
            log_prog('Convert yn_tf columns to binary')
            self.__dataset = self.__convert_yn_tf_to_binary(self.__dataset, yn_tf_cols)
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')

    @staticmethod
    def __clean_data(original_dataset: pd.DataFrame = pd.DataFrame(), handle_missing: str = 'leave') -> pd.DataFrame:
        log_prog('Enter classes/DataWranglerClass.clean_data')
        if handle_missing == 'leave' or original_dataset.empty:
            log_prog('Exit classes/DataWranglerClass.clean_data')
            return original_dataset
        log_err('Only "leave" case is supported out of ' + ', '.join(DataWranglerClass.__handle_missing_options) + ' AS OF NOW')
        log_prog('Exit classes/DataWranglerClass.clean_data')
        exit()

    @staticmethod
    def __one_hot_encoder(original_dataset: pd.DataFrame = pd.DataFrame(), cols_to_encode: Union[list[str], None] = []) -> pd.DataFrame:
        log_prog('Enter classes/DataWranglerClass.one_hot_encoder')
        log_prog('Perform parameter pre-checks')
        if cols_to_encode is None:
            log_prog('No columns to hot-encode')
            return original_dataset
        if original_dataset.empty:
            log_err('Please provide the dataset which needs to be one-hot encoded')
            return pd.DataFrame()
        if len(cols_to_encode) == 0:
            log_err('Please provide columns which are to be one-hot encoded, or pass "None"')
            return pd.DataFrame()
        for column in cols_to_encode:
            if column not in original_dataset.columns:
                log_err('Column to encode "' + str(column) + '" is absent in the data present in the file')
                return pd.DataFrame()
        log_prog('Complete parameter pre-checks')
        log_prog('One-hot encode provided columns')
        modified_dataset: pd.DataFrame = original_dataset.copy()
        new_columns_for_modified_dataset: dict[str, pd.Series] = {}
        for column in cols_to_encode:
            log_prog('One-hot encode column "' + str(column) + '"')
            unique_values = original_dataset[column].unique()
            unique_values.sort()
            new_columns = [column + "_is_" + str(x) for x in unique_values]
            for new_column in new_columns:
                new_columns_for_modified_dataset[new_column] = original_dataset[column].apply(lambda x: 1 if str(x) == new_column[len(column) + 4:] else 0)
            del modified_dataset[column]

        modified_dataset = modified_dataset.assign(**new_columns_for_modified_dataset)
        log_prog('Exit classes/DataWranglerClass.one_hot_encoder')
        return modified_dataset

    @staticmethod
    def __convert_yn_tf_to_binary(original_dataset: pd.DataFrame = pd.DataFrame(), cols_to_encode: Union[list[str], None] = []) -> pd.DataFrame:
        log_prog('Enter classes/DataWranglerClass.convert_yn_tf_to_binary')
        log_prog('Perform parameter pre-checks')
        if cols_to_encode is None:
            log_prog('No columns to encode')
            return
        if original_dataset.empty:
            log_err('Please provide the dataset which needs to be wrangled')
            return pd.DataFrame()
        if len(cols_to_encode) == 0:
            log_err('Please provide columns which are to be one-hot encoded, or pass "None"')
            return pd.DataFrame()
        for column in cols_to_encode:
            if column not in original_dataset.columns:
                log_err('Column to encode "' + str(column) + '" is absent in the data present in the file')
                return pd.DataFrame()
        log_prog('Complete parameter pre-checks')
        modified_dataset: pd.DataFrame = original_dataset.copy()
        for column in cols_to_encode:
            log_prog('Encode column "' + str(column) + '" in binary')
            modified_dataset[column].replace(['True', 'T', 'true', 't', 'yes', 'Yes', 'Y', 'y', '1'], 1, inplace = True)
            modified_dataset[column].replace(['False', 'F', 'false', 'f', 'no', 'No', 'N', 'n', '0'], 0, inplace = True)
        log_prog('Exit classes/DataWranglerClass.convert_yn_tf_to_binary')
        return modified_dataset

    def get_processed_dataframe(self) -> pd.DataFrame:
        return self.__dataset
