'''

TODO: Document the file-functioning well.

Data Wrangler:
- Implements
    - Data Cleaning
    - Data One-Shot Encoding
- Methods
    - Binary encoding yN, tF columns

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

    def __init__(self, file_path: str, handle_missing: str = 'leave', cols_to_encode: Union[list[str], None] = []):
        log_prog('Enter classes/' + type(self).__name__ + '.constructor')

        log_prog('Perform parameter pre-checks')
        premature_return: bool = False
        if handle_missing not in self.__handle_missing_options:
            log_err('Invalid handle_missing option provided')
            premature_return = True
        if cols_to_encode is not None and len(cols_to_encode) == 0:
            log_err('Please provide columns which are to be one-hot encoded, or pass "None"')
            premature_return = True
        if premature_return:
            self.__dataset = []
            return
        log_prog('Complete parameter pre-checks')

        log_prog('Retrieve data from provided location')
        file_extension: str = os.path.splitext(file_path)[1]
        self.__dataset: pd.DataFrame = self.__readers[file_extension](file_path)

        log_prog('Clean data')
        self.__dataset = self.__clean_data(handle_missing)

        if cols_to_encode is not None:
            log_prog('One-hot encode data')
            self.__dataset = self.__one_hot_encoder(cols_to_encode)
        log_prog('Exit classes/' + type(self).__name__ + '.constructor')

    def __clean_data(self, handle_missing: str) -> pd.DataFrame:
        log_prog('Enter classes/' + type(self).__name__ + '.clean_data')
        if handle_missing == 'leave':
            return self.__dataset
        log_err('TODO: Handle for each case in' + str(self.__handle_missing_options))
        log_prog('Exit classes/' + type(self).__name__ + '.clean_data')

    def __one_hot_encoder(self, cols_to_encode: Union[list[str], None] = []) -> pd.DataFrame:
        log_prog('Enter classes/' + type(self).__name__ + '.one_hot_encoder')
        log_prog('Perform parameter pre-checks')
        if cols_to_encode is None:
            log_prog('No columns to hot-encode')
            return self.__dataset
        if len(cols_to_encode) == 0:
            log_err('Please provide columns which are to be one-hot encoded, or pass "None"')
            return []
        for column in cols_to_encode:
            if column not in self.__dataset.columns:
                log_err('Column to encode "' + str(column) + '" is absent in the data present in the file')
                return []
        log_prog('Complete parameter pre-checks')
        log_prog('One-hot encode provided columns')
        for column in cols_to_encode:
            unique_values = self.__dataset[column].unique()
            unique_values.sort()
            new_columns = [column + "_is_" + str(x) for x in unique_values]

            for new_column in new_columns:
                self.__dataset[new_column] = self.__dataset[column].apply(lambda x: 1 if str(x) == new_column[len(column) + 4:] else 0)
            del self.__dataset[column]

        # TODO: Check if there is any entry in dataset that is not a number
        log_prog('Exit classes/' + type(self).__name__ + '.one_hot_encoder')
        return self.__dataset

    def convert_yn_tf_to_binary(self, cols_to_encode: Union[list[str], None] = []) -> None:
        log_prog('Enter classes/' + type(self).__name__ + '.convert_yn_tf_to_binary')
        log_prog('Perform parameter pre-checks')
        if cols_to_encode is None:
            log_prog('No columns to encode')
            return
        if len(cols_to_encode) == 0:
            log_err('Please provide columns which are to be one-hot encoded, or pass "None"')
            return
        for column in cols_to_encode:
            if column not in self.__dataset.columns:
                log_err('Column to encode "' + str(column) + '" is absent in the data present in the file')
                return
        log_prog('Complete parameter pre-checks')
        for column in cols_to_encode:
            log_prog('Encode column "' + str(column) + '"')
            self.__dataset[column].replace(['True', 'T', 'true', 't', 'yes', 'Yes', 'Y', 'y', '1'], 1, inplace = True)
            self.__dataset[column].replace(['False', 'F', 'false', 'f', 'no', 'No', 'N', 'n', '0'], 0, inplace = True)
        log_prog('Exit classes/' + type(self).__name__ + '.convert_yn_tf_to_binary')

    def get_processed_dataframe(self) -> pd.DataFrame:
        return self.__dataset
