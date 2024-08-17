# Class Name - FairMetric
# Description: This class is used to calculate the fairness metric of the model (Using the paper - Fairness definitions explained by Sahil et al)
# Following methods -
# 1. __init__ - Constructor (Initializes the class variables)
# 2. 


from typing import List, Union
import logging

logging.getLogger('requests').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO) # Comment this line to stop showing the messages

class FairMetricClass:

    def __init__(self, classification_output: list[tuple[list[any], any, any]],  # The output of the classifier includes X_test, y_test, y_pred
                sensitive_attribute_index: int,  # The index of the sensitive attribute in the dataset
                sensitive_attribute_value: int,  # The value of the sensitive attribute (assuming binary sensitive attribute)
                ) -> None:
                
        if sensitive_attribute_index >= len(classification_output[0][0]) :
            logging.error('Invalid sensitive attribute index')
            return
        
        self.__classification_output: list[tuple[list[any], any, any]] = classification_output
        self.__sensitive_attribute_index: int = sensitive_attribute_index
        self.__sensitive_attribute_value: int = sensitive_attribute_value


        # Segregate classification output based on the sensitive attribute (One will be the protected group with the given sensitive attribute value and the other will be the unprotected group)
        self.__protected_group: list[tuple[list[any], any, any]] = []
        self.__unprotected_group: list[tuple[list[any], any, any]] = []
        for j in range(len(self.__classification_output)):
            if self.__classification_output[j][0][self.__sensitive_attribute_index] == self.__sensitive_attribute_value:
                self.__protected_group.append(self.__classification_output[j])
            else:
                self.__unprotected_group.append(self.__classification_output[j])

        # Calculate confusion matrices
        self.__protected_confusion_mat = self.__create_confusion_matrix(self.__protected_group)
        self.__unprotected_confusion_mat = self.__create_confusion_matrix(self.__unprotected_group)

        logging.info("************Fairness Metric************")
        logging.info('Sensitive Attribute Index: {}'.format(self.__sensitive_attribute_index))
        logging.info('Sensitive Attribute Value: {}'.format(self.__sensitive_attribute_value))
        logging.info('Protected Group -> {}'.format(self.__protected_group))
        logging.info('Unprotected Group -> {}'.format(self.__unprotected_group))
        logging.debug('*********** Protected Group confusion matrix ***********')
        self.__display_confusion_matrix(self.__protected_confusion_mat)
        logging.debug('*********** Unprotected Group confusion matrix ***********')
        self.__display_confusion_matrix(self.__unprotected_confusion_mat)
        logging.debug("**************************************************")

    # Method to create confusion matrix for both protected and unprotected groups (Not valid for multi-label classification)
    def __create_confusion_matrix(self, group: list[tuple[list[any], any, any]]) -> list[list[int]]:
        confusion_matrix: List[List[int]] = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
        for i in range(len(group)):
            if group[i][1] == 0 and group[i][2] == 0:
                confusion_matrix[0][0] += 1
            elif group[i][1] == 0 and group[i][2] == 1:
                confusion_matrix[0][1] += 1
            elif group[i][1] == 1 and group[i][2] == 0:
                confusion_matrix[1][0] += 1
            elif group[i][1] == 1 and group[i][2] == 1:
                confusion_matrix[1][1] += 1
        return confusion_matrix
    
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
        ) -> None:

        print(resolving_feature_index)
        for metric in metrics_list:
            print(metric)

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
        logging.info("*******************************************")

        return [protected_group_acceptance_rate == unprotected_group_acceptance_rate, {"accepted_rate_diff": abs(protected_group_acceptance_rate - unprotected_group_acceptance_rate)}]

    def check_conditional_parity(self, resolving_feature_index: int, resolving_feature_value: any) -> tuple[bool, dict]:
        # Check for protected group
        print("************Conditional Parity************")

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

        print('Protected Group Conditional Acceptance Rate: ', protected_group_conditional_acceptance_rate)
        print('Unprotected Group Conditional Acceptance Rate: ', unprotected_group_conditional_acceptance_rate)
        print("******************************************")

        return [protected_group_conditional_acceptance_rate == unprotected_group_conditional_acceptance_rate, {"conditional_acc_rate_diff": abs(protected_group_conditional_acceptance_rate - unprotected_group_conditional_acceptance_rate)}]

    def check_equalized_odds(self) -> tuple[bool, dict]:
        print("************Equalized Odds************")
        # Calculate true positive rate and false positive rate for both groups
        protected_group_true_positive_rate: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][0] + self.__protected_confusion_mat[1][1])
        protected_group_false_positive_rate: float = self.__divide(self.__protected_confusion_mat[0][1], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[0][1])
        unprotected_group_true_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][0] + self.__unprotected_confusion_mat[1][1])
        unprotected_group_false_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[0][1], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[0][1])

        print('Protected Group True Positive Rate: ', protected_group_true_positive_rate)
        print('Protected Group False Positive Rate: ', protected_group_false_positive_rate)
        print('Unprotected Group True Positive Rate: ', unprotected_group_true_positive_rate)
        print('Unprotected Group False Positive Rate: ', unprotected_group_false_positive_rate)
        print("***************************************")

        return [protected_group_true_positive_rate == unprotected_group_true_positive_rate and protected_group_false_positive_rate == unprotected_group_false_positive_rate, {"true_positive_rate_diff": abs(protected_group_true_positive_rate - unprotected_group_true_positive_rate), "false_positive_rate_diff": abs(protected_group_false_positive_rate - unprotected_group_false_positive_rate)}]

    def check_equal_opportunity(self) -> tuple[bool, dict]:
        print("************Equal Opportunity************")

        # Calculate true positive rate and false positive rate for both groups
        protected_group_true_positive_rate: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][0] + self.__protected_confusion_mat[1][1])
        unprotected_group_true_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][0] + self.__unprotected_confusion_mat[1][1])

        print('Protected Group True Positive Rate: ', protected_group_true_positive_rate)
        print('Unprotected Group True Positive Rate: ', unprotected_group_true_positive_rate)
        print("***************************************")

        return [protected_group_true_positive_rate == unprotected_group_true_positive_rate, {"true_positive_rate_diff": abs(protected_group_true_positive_rate - unprotected_group_true_positive_rate)}]

    def check_predictive_equality(self) -> tuple[bool, dict]:
        print("************Predictive Equality************")

        # Calculate true positive rate and false positive rate for both groups
        protected_group_false_positive_rate: float = self.__divide(self.__protected_confusion_mat[0][1], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[0][1])
        unprotected_group_false_positive_rate: float = self.__divide(self.__unprotected_confusion_mat[0][1], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[0][1])

        print('Protected Group False Positive Rate: ', protected_group_false_positive_rate)
        print('Unprotected Group False Positive Rate: ', unprotected_group_false_positive_rate)
        print("********************************************")

        return [protected_group_false_positive_rate == unprotected_group_false_positive_rate, {"false_positive_rate_diff": abs(protected_group_false_positive_rate - unprotected_group_false_positive_rate)}]

    def check_conditional_use_accuracy_equality(self) -> tuple[bool, dict]:
        print("************Conditional Use Accuracy Equality************")
        
        # Calculate positive and negative predictive values for both groups
        protected_group_ppv: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][1] + self.__protected_confusion_mat[0][1])
        unprotected_group_ppv: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][1] + self.__unprotected_confusion_mat[0][1])
        protected_group_npv: float = self.__divide(self.__protected_confusion_mat[0][0], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[1][0])
        unprotected_group_npv: float = self.__divide(self.__unprotected_confusion_mat[0][0], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[1][0])

        print('Protected Group Positive Predictive Value: ', protected_group_ppv)
        print('Protected Group Negative Predictive Value: ', protected_group_npv)
        print('Unprotected Group Positive Predictive Value: ', unprotected_group_ppv)
        print('Unprotected Group Negative Predictive Value: ', unprotected_group_npv)
        print("*********************************************************")

        return [protected_group_ppv == unprotected_group_ppv and protected_group_npv == unprotected_group_npv, {"ppv_diff": abs(protected_group_ppv - unprotected_group_ppv), "npv_diff": abs(protected_group_npv - unprotected_group_npv)}]

    def check_predictive_parity(self) -> tuple[bool, dict]:
        print("************Predictive Parity****************************")

        # Calculate positive and negative predictive values for both groups
        protected_group_ppv: float = self.__divide(self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[1][1] + self.__protected_confusion_mat[0][1])
        unprotected_group_ppv: float = self.__divide(self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[1][1] + self.__unprotected_confusion_mat[0][1])

        print('Protected Group Positive Predictive Value: ', protected_group_ppv)
        print('Unprotected Group Positive Predictive Value: ', unprotected_group_ppv)
        print("*********************************************************")

        return [protected_group_ppv == unprotected_group_ppv, {"ppv_diff": abs(protected_group_ppv - unprotected_group_ppv)}]

    def check_overall_accuracy_equality(self) -> tuple[bool, dict]:
        print("************Overall Accuracy Equality************")

        # Calculate overall accuracy for both groups
        protected_group_overall_accuracy: float = self.__divide(self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[1][1], self.__protected_confusion_mat[0][0] + self.__protected_confusion_mat[0][1] + self.__protected_confusion_mat[1][0] + self.__protected_confusion_mat[1][1])
        unprotected_group_overall_accuracy: float = self.__divide(self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[1][1], self.__unprotected_confusion_mat[0][0] + self.__unprotected_confusion_mat[0][1] + self.__unprotected_confusion_mat[1][0] + self.__unprotected_confusion_mat[1][1])

        print('Protected Group Overall Accuracy: ', protected_group_overall_accuracy)
        print('Unprotected Group Overall Accuracy: ', unprotected_group_overall_accuracy)
        print("************************************************")

        return [protected_group_overall_accuracy == unprotected_group_overall_accuracy, {"OverallAccuracyDiff": abs(protected_group_overall_accuracy - unprotected_group_overall_accuracy)}]
