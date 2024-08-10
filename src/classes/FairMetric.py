# Class Name - FairMetric
# Description: This class is used to calculate the fairness metric of the model (Using the paper - Fairness definitions explained by Sahil et al)
# Following methods -
# 1. __init__ - Constructor (Initializes the class variables)
# 2. fairnessMetric - Calculates the fairness metric
# Following variables -
from typing import List


class FairMetric:

    def __init__(self, classification_output: List[List[any], any, any],  # The output of the classifier includes X_test, y_test, y_pred
                 sensitive_attribute_index: int,  # The index of the sensitive attribute in the dataset
                 sensitive_attribute_value: int,  # The value of the sensitive attribute (assuming binary sensitive attribute)
                 ) -> None:
        self.__classification_output: List[List[any], any, any] = classification_output
        self.__sensitive_attribute_index: int = sensitive_attribute_index
        self.__sensitive_attribute_value: int = sensitive_attribute_value

        print("************Fairness Metric************")
        print('Sensitive Attribute Index: ', self.__sensitive_attribute_index)
        print('Sensitive Attribute Value: ', self.__sensitive_attribute_value)
        print("****************************************")

        # Segregate classification output based on the sensitive attribute (One will be the protected group with the given sensitive attribute value and the other will be the unprotected group)
        self.__protected_group: List[List[any], any, any] = []
        self.__unprotected_group: List[List[any], any, any] = []
        for j in range(len(self.__classification_output)):
            if self.__classification_output[j][0][self.__sensitive_attribute_index] == self.__sensitive_attribute_value:
                self.__protected_group.append(self.__classification_output[j])
            else:
                self.__unprotected_group.append(self.__classification_output[j])

    # Method to create confusion matrix for both protected and unprotected groups (Not valid for multi-label classification)
    def __create_confusion_matrix(self, group: List[List[any], any, any]) -> List[List[int]]:
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

    def check_statistical_parity(self) -> List[bool, dict]:
        # Confusion matrix calculation for protected and unprotected groups
        protected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__protected_group)
        unprotected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__unprotected_group)

        # Statistical parity check
        protected_group_acceptance_rate: float = (protected_group_confusion_matrix[1][1] + protected_group_confusion_matrix[0][1]) / (protected_group_confusion_matrix[0][0] + protected_group_confusion_matrix[0][1] + protected_group_confusion_matrix[1][0] + protected_group_confusion_matrix[1][1])
        unprotected_group_acceptance_rate: float = (unprotected_group_confusion_matrix[1][1] + unprotected_group_confusion_matrix[0][1]) / (unprotected_group_confusion_matrix[0][0] + unprotected_group_confusion_matrix[0][1] + unprotected_group_confusion_matrix[1][0] + unprotected_group_confusion_matrix[1][1])

        print("************Statistical Parity************")
        print('Protected Group Acceptance Rate: ', protected_group_acceptance_rate)
        print('Unprotected Group Acceptance Rate: ', unprotected_group_acceptance_rate)
        print("*******************************************")

        return [protected_group_acceptance_rate == unprotected_group_acceptance_rate, {"accepted_rate_diff": abs(protected_group_acceptance_rate - unprotected_group_acceptance_rate)}]

    def check_conditional_parity(self, resolving_feature_index: int, resolving_feature_value: any) -> List[bool, dict]:
        # Check for protected group
        protected_group_resolving_feature_count: int = 0
        protected_group_resolving_feature_positive_count: int = 0
        for i in range(len(self.__protected_group)):
            if self.__protected_group[i][0][resolving_feature_index] == resolving_feature_value:
                protected_group_resolving_feature_count += 1
                if self.__protected_group[i][2] == 1:
                    protected_group_resolving_feature_positive_count += 1
        protected_group_conditional_acceptance_rate: float = protected_group_resolving_feature_positive_count / protected_group_resolving_feature_count

        # Check for unprotected group
        unprotected_group_resolving_feature_count: int = 0
        unprotected_group_resolving_feature_positive_count: int = 0
        for i in range(len(self.__unprotected_group)):
            if self.__unprotected_group[i][0][resolving_feature_index] == resolving_feature_value:
                unprotected_group_resolving_feature_count += 1
                if self.__unprotected_group[i][2] == 1:
                    unprotected_group_resolving_feature_positive_count += 1
        unprotected_group_conditional_acceptance_rate: float = unprotected_group_resolving_feature_positive_count / unprotected_group_resolving_feature_count

        print("************Conditional Parity************")
        print('Protected Group Conditional Acceptance Rate: ', protected_group_conditional_acceptance_rate)
        print('Unprotected Group Conditional Acceptance Rate: ', unprotected_group_conditional_acceptance_rate)
        print("******************************************")

        return [protected_group_conditional_acceptance_rate == unprotected_group_conditional_acceptance_rate, {"conditional_acc_rate_diff": abs(protected_group_conditional_acceptance_rate - unprotected_group_conditional_acceptance_rate)}]

    def check_equalized_odds(self) -> List[bool, dict]:
        # Calculate confusion matrix for protected and unprotected groups
        protected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__protected_group)
        unprotected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__unprotected_group)

        # Calculate true positive rate and false positive rate for both groups
        protected_group_true_positive_rate: float = protected_group_confusion_matrix[1][1] / (protected_group_confusion_matrix[1][0] + protected_group_confusion_matrix[1][1])
        protected_group_false_positive_rate: float = protected_group_confusion_matrix[0][1] / (protected_group_confusion_matrix[0][0] + protected_group_confusion_matrix[0][1])
        unprotected_group_true_positive_rate: float = unprotected_group_confusion_matrix[1][1] / (unprotected_group_confusion_matrix[1][0] + unprotected_group_confusion_matrix[1][1])
        unprotected_group_false_positive_rate: float = unprotected_group_confusion_matrix[0][1] / (unprotected_group_confusion_matrix[0][0] + unprotected_group_confusion_matrix[0][1])

        print("************Equalized Odds************")
        print('Protected Group True Positive Rate: ', protected_group_true_positive_rate)
        print('Protected Group False Positive Rate: ', protected_group_false_positive_rate)
        print('Unprotected Group True Positive Rate: ', unprotected_group_true_positive_rate)
        print('Unprotected Group False Positive Rate: ', unprotected_group_false_positive_rate)
        print("***************************************")

        return [protected_group_true_positive_rate == unprotected_group_true_positive_rate and protected_group_false_positive_rate == unprotected_group_false_positive_rate, {"true_positive_rate_diff": abs(protected_group_true_positive_rate - unprotected_group_true_positive_rate), "false_positive_rate_diff": abs(protected_group_false_positive_rate - unprotected_group_false_positive_rate)}]

    def check_equal_opportunity(self) -> List[bool, dict]:
        # Calculate confusion matrix for protected and unprotected groups
        protected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__protected_group)
        unprotected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__unprotected_group)

        # Calculate true positive rate and false positive rate for both groups
        protected_group_true_positive_rate: float = protected_group_confusion_matrix[1][1] / (protected_group_confusion_matrix[1][0] + protected_group_confusion_matrix[1][1])
        unprotected_group_true_positive_rate: float = unprotected_group_confusion_matrix[1][1] / (unprotected_group_confusion_matrix[1][0] + unprotected_group_confusion_matrix[1][1])

        print("************Equal Opportunity************")
        print('Protected Group True Positive Rate: ', protected_group_true_positive_rate)
        print('Unprotected Group True Positive Rate: ', unprotected_group_true_positive_rate)
        print("***************************************")

        return [protected_group_true_positive_rate == unprotected_group_true_positive_rate, {"true_positive_rate_diff": abs(protected_group_true_positive_rate - unprotected_group_true_positive_rate)}]

    def check_predictive_equality(self) -> List[bool, dict]:
        # Calculate confusion matrix for protected and unprotected groups
        protected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__protected_group)
        unprotected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__unprotected_group)

        # Calculate true positive rate and false positive rate for both groups
        protected_group_false_positive_rate: float = protected_group_confusion_matrix[0][1] / (protected_group_confusion_matrix[0][0] + protected_group_confusion_matrix[0][1])
        unprotected_group_false_positive_rate: float = unprotected_group_confusion_matrix[0][1] / (unprotected_group_confusion_matrix[0][0] + unprotected_group_confusion_matrix[0][1])

        print("************Predictive Equality************")
        print('Protected Group False Positive Rate: ', protected_group_false_positive_rate)
        print('Unprotected Group False Positive Rate: ', unprotected_group_false_positive_rate)
        print("********************************************")

        return [protected_group_false_positive_rate == unprotected_group_false_positive_rate, {"false_positive_rate_diff": abs(protected_group_false_positive_rate - unprotected_group_false_positive_rate)}]

    def check_conditional_use_accuracy_equality(self) -> List[bool, dict]:
        # Calculate confusion matrix for protected and unprotected groups
        protected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__protected_group)
        unprotected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__unprotected_group)

        # Calculate positive and negative predictive values for both groups
        protected_group_ppv: float = protected_group_confusion_matrix[1][1] / (protected_group_confusion_matrix[1][1] + protected_group_confusion_matrix[0][1])
        unprotected_group_ppv: float = unprotected_group_confusion_matrix[1][1] / (unprotected_group_confusion_matrix[1][1] + unprotected_group_confusion_matrix[0][1])
        protected_group_npv: float = protected_group_confusion_matrix[0][0] / (protected_group_confusion_matrix[0][0] + protected_group_confusion_matrix[1][0])
        unprotected_group_npv: float = unprotected_group_confusion_matrix[0][0] / (unprotected_group_confusion_matrix[0][0] + unprotected_group_confusion_matrix[1][0])

        print("************Conditional Use Accuracy Equality************")
        print('Protected Group Positive Predictive Value: ', protected_group_ppv)
        print('Protected Group Negative Predictive Value: ', protected_group_npv)
        print('Unprotected Group Positive Predictive Value: ', unprotected_group_ppv)
        print('Unprotected Group Negative Predictive Value: ', unprotected_group_npv)
        print("*********************************************************")

        return [protected_group_ppv == unprotected_group_ppv and protected_group_npv == unprotected_group_npv, {"ppv_diff": abs(protected_group_ppv - unprotected_group_ppv), "npv_diff": abs(protected_group_npv - unprotected_group_npv)}]

    def check_predictive_parity(self) -> List[bool, dict]:
        # Calculate confusion matrix for protected and unprotected groups
        protected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__protected_group)
        unprotected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__unprotected_group)

        # Calculate positive and negative predictive values for both groups
        protected_group_ppv: float = protected_group_confusion_matrix[1][1] / (protected_group_confusion_matrix[1][1] + protected_group_confusion_matrix[0][1])
        unprotected_group_ppv: float = unprotected_group_confusion_matrix[1][1] / (unprotected_group_confusion_matrix[1][1] + unprotected_group_confusion_matrix[0][1])

        print("************Predictive Parity****************************")
        print('Protected Group Positive Predictive Value: ', protected_group_ppv)
        print('Unprotected Group Positive Predictive Value: ', unprotected_group_ppv)
        print("*********************************************************")

        return [protected_group_ppv == unprotected_group_ppv, {"ppv_diff": abs(protected_group_ppv - unprotected_group_ppv)}]

    def check_overall_accuracy_equality(self) -> List[bool, dict]:
        # Calculate confusion matrix for protected and unprotected groups
        protected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__protected_group)
        unprotected_group_confusion_matrix: List[List[int]] = self.__create_confusion_matrix(self.__unprotected_group)

        # Calculate overall accuracy for both groups
        protected_group_overall_accuracy: float = (protected_group_confusion_matrix[0][0] + protected_group_confusion_matrix[1][1]) / (protected_group_confusion_matrix[0][0] + protected_group_confusion_matrix[0][1] + protected_group_confusion_matrix[1][0] + protected_group_confusion_matrix[1][1])
        unprotected_group_overall_accuracy: float = (unprotected_group_confusion_matrix[0][0] + unprotected_group_confusion_matrix[1][1]) / (unprotected_group_confusion_matrix[0][0] + unprotected_group_confusion_matrix[0][1] + unprotected_group_confusion_matrix[1][0] + unprotected_group_confusion_matrix[1][1])

        print("************Overall Accuracy Equality************")
        print('Protected Group Overall Accuracy: ', protected_group_overall_accuracy)
        print('Unprotected Group Overall Accuracy: ', unprotected_group_overall_accuracy)
        print("************************************************")

        return [protected_group_overall_accuracy == unprotected_group_overall_accuracy, {"OverallAccuracyDiff": abs(protected_group_overall_accuracy - unprotected_group_overall_accuracy)}]
