# Class Name - FairMetric
# Description: This class is used to calculate the fairness metric of the model (Using the paper - Fairness definitions explained by Sahil et al)
# Following methods -
# 1. __init__ - Constructor (Initializes the class variables)
# 2.


from typing import List, Union
import pandas as pd
import logging
from sklearn.metrics import confusion_matrix

logging.getLogger('requests').setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG, format="%(message)s\n") # Comment this line to stop showing the messages

class FairMetricClass:

    def __init__(self, classification_output: pd.DataFrame,  # The output of the classifier includes X_test, y_test, y_pred
                sensitive_attribute_column: str,  # The name of the sensitive attribute column in the dataset
                sensitive_attribute_value: int,  # The value of the sensitive attribute (assuming binary sensitive attribute)
                target_column_name: str,
                pred_column_name: str
            ) -> None:

        if classification_output[sensitive_attribute_column] is None:
            logging.error('Sensitive attribute column not found')
            return

        self.__classification_output: pd.DataFrame = classification_output
        self.__sensitive_attribute_column: str = sensitive_attribute_column
        self.__sensitive_attribute_value: int = sensitive_attribute_value
        self.__target_column_name: str = target_column_name
        self.__pred_column_name: str = pred_column_name


        # Segregate classification output based on the sensitive attribute (One will be the protected group with the given sensitive attribute value and the other will be the unprotected group)
        self.__protected_group: list[tuple[list[any], any, any]] = [
            [row.tolist(), row[self.__target_column_name], row[self.__pred_column_name]] 
            for index, row in self.__classification_output.iterrows() if row[self.__sensitive_attribute_column] == self.__sensitive_attribute_value
        ]
        self.__unprotected_group: list[tuple[list[any], any, any]] = [
            [row.tolist(), row[self.__target_column_name], row[self.__pred_column_name]]
            for index, row in self.__classification_output.iterrows() if row[self.__sensitive_attribute_column] != self.__sensitive_attribute_value
        ]
        

        # Calculate confusion matrices
        self.__protected_confusion_mat = self.__create_confusion_matrix(self.__protected_group)
        self.__unprotected_confusion_mat = self.__create_confusion_matrix(self.__unprotected_group)

        logging.info("************Fairness Metric initialization************")
        logging.info('Sensitive Attribute Column: {}'.format(self.__sensitive_attribute_column))
        logging.info('Sensitive Attribute Value: {}'.format(self.__sensitive_attribute_value))
        logging.info('Protected Group -> {}'.format(self.__protected_group))
        logging.info('Unprotected Group -> {}'.format(self.__unprotected_group))
        logging.debug('*********** Protected Group confusion matrix ***********')
        self.__display_confusion_matrix(self.__protected_confusion_mat)
        logging.debug('*********** Unprotected Group confusion matrix ***********')
        self.__display_confusion_matrix(self.__unprotected_confusion_mat)
        logging.debug("*************Initialization Finish*********************\n\n")

    # Method to create confusion matrix for both protected and unprotected groups (Not valid for multi-label classification)
    def __create_confusion_matrix(self, group: list[tuple[list[any], any, any]]) -> list[list[int]]:
        y_true = [row[1] for row in group]
        y_pred = [row[2] for row in group]
        confusion_mat = confusion_matrix(y_true, y_pred)
        confusion_mat_total = sum(sum(row) for row in confusion_mat)
        print(confusion_mat)
        binary_confusion_matrix: List[List[int]] = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
        for k in range (len(confusion_mat)):
            # Get TP from confusion matrix
            binary_confusion_matrix[1][1] += confusion_mat[k][k]
            # Get FP from confusion matrix
            binary_confusion_matrix[0][1] += sum(confusion_mat[k]) - confusion_mat[k][k]
            # Get FN from confusion matrix
            binary_confusion_matrix[1][0] += sum(confusion_mat[i][k] for i in range(len(confusion_mat)) if i != k)
            # Get TN from confusion matrix
            binary_confusion_matrix[0][0] += confusion_mat_total - binary_confusion_matrix[0][1] - binary_confusion_matrix[1][0] - binary_confusion_matrix[1][1]

        print("******************************************")
        return binary_confusion_matrix

    def __display_confusion_matrix(self, confusion_mat: list[list[int]]) -> None:
        # Assuming confusion_matrix is in the form [[TN, FP], [FN, TP]]
        logging.debug(f"True Negative: {confusion_mat[0][0]}  | False Positive: {confusion_mat[0][1]}\n")
        logging.debug(f"False Negative: {confusion_mat[1][0]} | True Positive : {confusion_mat[1][1]}\n")

    def __divide(self, num, den) -> float:
        if den == 0 :
            logging.error('Division by zero')
            return -1
        return num / den

    def check_multiple(
        self,
        metrics_list: Union[
            'statistical_parity',
            'conditional_parity',
            'equalized_odds',
            'equal_opportunity',
            'predictive_equality',
            'conditional_use_accuracy_equality',
            'predictive_parity',
            'overall_accuracy_equality'
        ],
        resolving_feature_index = -1,
        resolving_feature_value = -1
        ) -> dict:

        metric_results: dict = {}

        for metric in metrics_list:
            if metric != 'conditional_parity':
                metric_func_name = 'check_{}'.format(metric)
                metric_func_call = getattr(self, metric_func_name)()
                logging.debug(list(metric_func_call[-1].values()))
            else:
                if resolving_feature_index == -1:
                    logging.error('Please give valid resolving feature index for conditional parity')
                else:
                    metric_func_name = 'check_{}'.format(metric)
                    metric_func_call = getattr(self, metric_func_name)(resolving_feature_index, resolving_feature_value)
                    logging.debug(list(metric_func_call[-1].values()))
            if len(list(metric_func_call[-1].values())) == 1:
                metric_results[metric] = list(metric_func_call[-1].values())[0]
            else:
                metric_results[metric] = sum(list(metric_func_call[-1].values()))

        logging.info("******************************************")
        return metric_results
    
    def check_statistical_parity(self) -> tuple[bool, dict]:
        # Statistical parity check
        logging.info("************Statistical Parity************")
        protected_group_acceptance_rate: float = self.__divide(
                self.__protected_confusion_mat[1][1] + self.__protected_confusion_mat[0][1] ,
                self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[0][1]
                + self.__protected_confusion_mat[1][0] + self.__protected_confusion_mat[1][1]
            )
        unprotected_group_acceptance_rate: float = self.__divide(
                self.__unprotected_confusion_mat[1][1] + self.__unprotected_confusion_mat[0][1] ,
                self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[0][1]
                + self.__unprotected_confusion_mat[1][0] + self.__unprotected_confusion_mat[1][1]
            )

        logging.info('Protected Group Acceptance Rate: {}'.format(protected_group_acceptance_rate))
        logging.info('Unprotected Group Acceptance Rate: {}'.format(unprotected_group_acceptance_rate))

        return [protected_group_acceptance_rate == unprotected_group_acceptance_rate, {"accepted_rate_diff": abs(protected_group_acceptance_rate - unprotected_group_acceptance_rate)}]

    def check_conditional_parity(self, resolving_feature_index: int, resolving_feature_value: any) -> tuple[bool, dict]:
        # Check for protected group
        logging.info("************Conditional Parity************")

        protected_group_resolving_feature_count: int = 0
        protected_group_resolving_feature_positive_count: int = 0
        for i in range(len(self.__protected_group)):
            if self.__protected_group[i][0][resolving_feature_index] == resolving_feature_value:
                protected_group_resolving_feature_count += 1
                if self.__protected_group[i][2] == 1:
                    protected_group_resolving_feature_positive_count += 1
        protected_group_conditional_acceptance_rate: float = self.__divide(protected_group_resolving_feature_positive_count, protected_group_resolving_feature_count)

        # Check for unprotected group
        unprotected_group_resolving_feature_count: int = 0
        unprotected_group_resolving_feature_positive_count: int = 0
        for i in range(len(self.__unprotected_group)):
            if self.__unprotected_group[i][0][resolving_feature_index] == resolving_feature_value:
                unprotected_group_resolving_feature_count += 1
                if self.__unprotected_group[i][2] == 1:
                    unprotected_group_resolving_feature_positive_count += 1
        unprotected_group_conditional_acceptance_rate: float = self.__divide(unprotected_group_resolving_feature_positive_count, unprotected_group_resolving_feature_count)

        logging.info('Protected Group Conditional Acceptance Rate: {}'.format(protected_group_conditional_acceptance_rate))
        logging.info('Unprotected Group Conditional Acceptance Rate: {}'.format(unprotected_group_conditional_acceptance_rate))
        logging.info("******************************************")

        return [protected_group_conditional_acceptance_rate == unprotected_group_conditional_acceptance_rate, {"conditional_acc_rate_diff": abs(protected_group_conditional_acceptance_rate - unprotected_group_conditional_acceptance_rate)}]

    def check_equalized_odds(self) -> tuple[bool, dict]:
        logging.info("************Equalized Odds************")
        # Calculate true positive rate and false positive rate for both groups
        protected_group_true_positive_rate: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][0] + self.__protected_confusion_mat[1][1])
        protected_group_false_positive_rate: float = self.__divide(self.__protected_confusion_mat[0][1], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[0][1])
        unprotected_group_true_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][0] + self.__unprotected_confusion_mat[1][1])
        unprotected_group_false_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[0][1], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[0][1])

        logging.info('Protected Group True Positive Rate: {}'.format(protected_group_true_positive_rate))
        logging.info('Protected Group False Positive Rate: {}'.format(protected_group_false_positive_rate))
        logging.info('Unprotected Group True Positive Rate: {}'.format(unprotected_group_true_positive_rate))
        logging.info('Unprotected Group False Positive Rate: {}'.format(unprotected_group_false_positive_rate))
        logging.info("***************************************")

        return [protected_group_true_positive_rate == unprotected_group_true_positive_rate and protected_group_false_positive_rate == unprotected_group_false_positive_rate, {"true_positive_rate_diff": abs(protected_group_true_positive_rate - unprotected_group_true_positive_rate), "false_positive_rate_diff": abs(protected_group_false_positive_rate - unprotected_group_false_positive_rate)}]

    def check_equal_opportunity(self) -> tuple[bool, dict]:
        logging.info("************Equal Opportunity************")

        # Calculate true positive rate and false positive rate for both groups
        protected_group_true_positive_rate: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][0] + self.__protected_confusion_mat[1][1])
        unprotected_group_true_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][0] + self.__unprotected_confusion_mat[1][1])

        logging.info('Protected Group True Positive Rate: {}'.format(protected_group_true_positive_rate))
        logging.info('Unprotected Group True Positive Rate: {}'.format(unprotected_group_true_positive_rate))
        logging.info("***************************************")

        return [protected_group_true_positive_rate == unprotected_group_true_positive_rate, {"true_positive_rate_diff": abs(protected_group_true_positive_rate - unprotected_group_true_positive_rate)}]

    def check_predictive_equality(self) -> tuple[bool, dict]:
        logging.info("************Predictive Equality************")

        # Calculate true positive rate and false positive rate for both groups
        protected_group_false_positive_rate: float = self.__divide(self.__protected_confusion_mat[0][1], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[0][1])
        unprotected_group_false_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[0][1], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[0][1])

        logging.info('Protected Group False Positive Rate: {}'.format(protected_group_false_positive_rate))
        logging.info('Unprotected Group False Positive Rate: {}'.format(unprotected_group_false_positive_rate))
        logging.info("********************************************")

        return [protected_group_false_positive_rate == unprotected_group_false_positive_rate, {"false_positive_rate_diff": abs(protected_group_false_positive_rate - unprotected_group_false_positive_rate)}]

    def check_conditional_use_accuracy_equality(self) -> tuple[bool, dict]:
        logging.info("************Conditional Use Accuracy Equality************")

        # Calculate positive and negative predictive values for both groups
        protected_group_ppv: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][1] + self.__protected_confusion_mat[0][1])
        unprotected_group_ppv: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][1] + self.__unprotected_confusion_mat[0][1])
        protected_group_npv: float = self.__divide(self.__protected_confusion_mat[0][0], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[1][0])
        unprotected_group_npv: float = self.__divide(self.__unprotected_confusion_mat[0][0], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[1][0])

        logging.info('Protected Group Positive Predictive Value: {}'.format(protected_group_ppv))
        logging.info('Protected Group Negative Predictive Value: {}'.format(protected_group_npv))
        logging.info('Unprotected Group Positive Predictive Value: {}'.format(unprotected_group_ppv))
        logging.info('Unprotected Group Negative Predictive Value: {}'.format(unprotected_group_npv))
        logging.info("*********************************************************")

        return [protected_group_ppv == unprotected_group_ppv and protected_group_npv == unprotected_group_npv, {"ppv_diff": abs(protected_group_ppv - unprotected_group_ppv), "npv_diff": abs(protected_group_npv - unprotected_group_npv)}]

    def check_predictive_parity(self) -> tuple[bool, dict]:
        logging.info("************Predictive Parity****************************")

        # Calculate positive and negative predictive values for both groups
        protected_group_ppv: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][1] + self.__protected_confusion_mat[0][1])
        unprotected_group_ppv: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][1] + self.__unprotected_confusion_mat[0][1])

        logging.info('Protected Group Positive Predictive Value: {}'.format(protected_group_ppv))
        logging.info('Unprotected Group Positive Predictive Value: {}'.format(unprotected_group_ppv))
        logging.info("*********************************************************")

        return [protected_group_ppv == unprotected_group_ppv, {"ppv_diff": abs(protected_group_ppv - unprotected_group_ppv)}]

    def check_overall_accuracy_equality(self) -> tuple[bool, dict]:
        logging.info("************Overall Accuracy Equality************")

        # Calculate overall accuracy for both groups
        protected_group_overall_accuracy: float = self.__divide(self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[0][1] + self.__protected_confusion_mat[1][0] + self.__protected_confusion_mat[1][1])
        unprotected_group_overall_accuracy: float = self.__divide(self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[0][1] + self.__unprotected_confusion_mat[1][0] + self.__unprotected_confusion_mat[1][1])

        logging.info('Protected Group Overall Accuracy: {}'.format(protected_group_overall_accuracy))
        logging.info('Unprotected Group Overall Accuracy: {}'.format(unprotected_group_overall_accuracy))
        logging.info("************************************************")

        return [protected_group_overall_accuracy == unprotected_group_overall_accuracy, {"OverallAccuracyDiff": abs(protected_group_overall_accuracy - unprotected_group_overall_accuracy)}]
