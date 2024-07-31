# Class Name - FairMetric
# Description: This class is used to calculate the fairness metric of the model (Using the paper - Fairness definitions explained by Sahil et al)
# Following methods -
# 1. __init__ - Constructor (Initializes the class variables)
# 2. fairnessMetric - Calculates the fairness metric
# Following variables -
from typing import List


class FairMetric:

    def __init__(
        self, classificationOutput: List[
            List[any], any, any
        ],  # The output of the classifier includes X_test, y_test, y_pred
        sensitiveAttributeIndex:
        int,  # The index of the sensitive attribute in the dataset
        sensitiveAttributeValue:
        int,  # The value of the sensitive attribute (assuming binary sensitive attribute)
    ) -> None:
        self.__classificationOutput: List[List[any], any,
                                          any] = classificationOutput
        self.__sensitiveAttributeIndex: int = sensitiveAttributeIndex
        self.__sensitiveAttributeValue: int = sensitiveAttributeValue

        print("************Fairness Metric************")

        print('Sensitive Attribute Index: ', self.__sensitiveAttributeIndex)
        print('Sensitive Attribute Value: ', self.__sensitiveAttributeValue)

        print("****************************************")
        # Segregate Classification Output based on the sensitive attribute (One will be the protected group with the given sensitive attribute value and the other will be the unprotected group)
        self.__protectedGroup: List[List[any], any, any] = []
        self.__unprotectedGroup: List[List[any], any, any] = []
        for j in range(len(self.__classificationOutput)):
            if self.__classificationOutput[j][0][
                self.__sensitiveAttributeIndex
            ] == self.__sensitiveAttributeValue:
                self.__protectedGroup.append(self.__classificationOutput[j])
            else:
                self.__unprotectedGroup.append(self.__classificationOutput[j])

    # Method to create confusion matrix for both protected and unprotected groups (Not valid for multi-label classification)
    def __createConfusionMatrix(
        self, group: List[List[any], any, any]
    ) -> List[List[int]]:
        confusionMatrix: List[List[int]] = [[0, 0],
                                            [0, 0]]  # [[TN, FP], [FN, TP]]
        for i in range(len(group)):
            if group[i][1] == 0 and group[i][2] == 0:
                confusionMatrix[0][0] += 1
            elif group[i][1] == 0 and group[i][2] == 1:
                confusionMatrix[0][1] += 1
            elif group[i][1] == 1 and group[i][2] == 0:
                confusionMatrix[1][0] += 1
            elif group[i][1] == 1 and group[i][2] == 1:
                confusionMatrix[1][1] += 1
        return confusionMatrix

    def checkStatisticalParity(self) -> bool:
        # Confusion matrix calculation for protected and unprotected groups
        protectedGroupConfusionMatrix: List[List[int]
                                            ] = self.__createConfusionMatrix(
                                                self.__protectedGroup
                                            )
        unprotectedGroupConfusionMatrix: List[List[int]
                                              ] = self.__createConfusionMatrix(
                                                  self.__unprotectedGroup
                                              )

        # Statistical Parity Check
        protectedGroupAcceptanceRate: float = (
            protectedGroupConfusionMatrix[1][1] +
            protectedGroupConfusionMatrix[0][1]
        ) / (
            protectedGroupConfusionMatrix[0][0] +
            protectedGroupConfusionMatrix[0][1] +
            protectedGroupConfusionMatrix[1][0] +
            protectedGroupConfusionMatrix[1][1]
        )
        unprotectedGroupAcceptanceRate: float = (
            unprotectedGroupConfusionMatrix[1][1] +
            unprotectedGroupConfusionMatrix[0][1]
        ) / (
            unprotectedGroupConfusionMatrix[0][0] +
            unprotectedGroupConfusionMatrix[0][1] +
            unprotectedGroupConfusionMatrix[1][0] +
            unprotectedGroupConfusionMatrix[1][1]
        )

        print("************Statistical Parity************")
        print(
            'Protected Group Acceptance Rate: ', protectedGroupAcceptanceRate
        )
        print(
            'Unprotected Group Acceptance Rate: ',
            unprotectedGroupAcceptanceRate
        )
        print("*******************************************")

        return protectedGroupAcceptanceRate == unprotectedGroupAcceptanceRate

    def checkConditionalParity(
        self, resolvingFeatureIndex: int, resolvingFeatureValue: any
    ) -> bool:
        # Check for protected group
        protectedGroupResolvingFeatureCount: int = 0
        protectedGroupResolvingFeaturePositiveCount: int = 0
        for i in range(len(self.__protectedGroup)):
            if self.__protectedGroup[i][0][resolvingFeatureIndex
                                           ] == resolvingFeatureValue:
                protectedGroupResolvingFeatureCount += 1
                if self.__protectedGroup[i][2] == 1:
                    protectedGroupResolvingFeaturePositiveCount += 1
        protectedGroupConditionalAcceptanceRate: float = protectedGroupResolvingFeaturePositiveCount / protectedGroupResolvingFeatureCount

        # Check for unprotected group
        unprotectedGroupResolvingFeatureCount: int = 0
        unprotectedGroupResolvingFeaturePositiveCount: int = 0
        for i in range(len(self.__unprotectedGroup)):
            if self.__unprotectedGroup[i][0][resolvingFeatureIndex
                                             ] == resolvingFeatureValue:
                unprotectedGroupResolvingFeatureCount += 1
                if self.__unprotectedGroup[i][2] == 1:
                    unprotectedGroupResolvingFeaturePositiveCount += 1
        unprotectedGroupConditionalAcceptanceRate: float = unprotectedGroupResolvingFeaturePositiveCount / unprotectedGroupResolvingFeatureCount

        print("************Conditional Parity************")
        print(
            'Protected Group Conditional Acceptance Rate: ',
            protectedGroupConditionalAcceptanceRate
        )
        print(
            'Unprotected Group Conditional Acceptance Rate: ',
            unprotectedGroupConditionalAcceptanceRate
        )
        print("******************************************")

        return protectedGroupConditionalAcceptanceRate == unprotectedGroupConditionalAcceptanceRate

    def checkEqualizedOdds(self) -> bool:
        # Calculate confusion matrix for protected and unprotected groups
        protectedGroupConfusionMatrix: List[List[int]
                                            ] = self.__createConfusionMatrix(
                                                self.__protectedGroup
                                            )
        unprotectedGroupConfusionMatrix: List[List[int]
                                              ] = self.__createConfusionMatrix(
                                                  self.__unprotectedGroup
                                              )

        # Calcuate True Positive Rate and False Positive Rate for both groups
        protectedGroupTruePositiveRate: float = protectedGroupConfusionMatrix[
            1][1] / (
                protectedGroupConfusionMatrix[1][0] +
                protectedGroupConfusionMatrix[1][1]
            )
        protectedGroupFalsePositiveRate: float = protectedGroupConfusionMatrix[
            0][1] / (
                protectedGroupConfusionMatrix[0][0] +
                protectedGroupConfusionMatrix[0][1]
            )
        unprotectedGroupTruePositiveRate: float = unprotectedGroupConfusionMatrix[
            1][1] / (
                unprotectedGroupConfusionMatrix[1][0] +
                unprotectedGroupConfusionMatrix[1][1]
            )
        unprotectedGroupFalsePositiveRate: float = unprotectedGroupConfusionMatrix[
            0][1] / (
                unprotectedGroupConfusionMatrix[0][0] +
                unprotectedGroupConfusionMatrix[0][1]
            )

        print("************Equalized Odds************")
        print(
            'Protected Group True Positive Rate: ',
            protectedGroupTruePositiveRate
        )
        print(
            'Protected Group False Positive Rate: ',
            protectedGroupFalsePositiveRate
        )
        print(
            'Unprotected Group True Positive Rate: ',
            unprotectedGroupTruePositiveRate
        )
        print(
            'Unprotected Group False Positive Rate: ',
            unprotectedGroupFalsePositiveRate
        )
        print("***************************************")

        return protectedGroupTruePositiveRate == unprotectedGroupTruePositiveRate and protectedGroupFalsePositiveRate == unprotectedGroupFalsePositiveRate

    def checkEqualOpportunity(self) -> bool:
        # Calculate confusion matrix for protected and unprotected groups
        protectedGroupConfusionMatrix: List[List[int]
                                            ] = self.__createConfusionMatrix(
                                                self.__protectedGroup
                                            )
        unprotectedGroupConfusionMatrix: List[List[int]
                                              ] = self.__createConfusionMatrix(
                                                  self.__unprotectedGroup
                                              )

        # Calcuate True Positive Rate and False Positive Rate for both groups
        protectedGroupTruePositiveRate: float = protectedGroupConfusionMatrix[
            1][1] / (
                protectedGroupConfusionMatrix[1][0] +
                protectedGroupConfusionMatrix[1][1]
            )
        unprotectedGroupTruePositiveRate: float = unprotectedGroupConfusionMatrix[
            1][1] / (
                unprotectedGroupConfusionMatrix[1][0] +
                unprotectedGroupConfusionMatrix[1][1]
            )

        print("************Equal Opportunity************")
        print(
            'Protected Group True Positive Rate: ',
            protectedGroupTruePositiveRate
        )
        print(
            'Unprotected Group True Positive Rate: ',
            unprotectedGroupTruePositiveRate
        )
        print("***************************************")

        return protectedGroupTruePositiveRate == unprotectedGroupTruePositiveRate

    def checkPredictiveEquality(self) -> bool:
        # Calculate confusion matrix for protected and unprotected groups
        protectedGroupConfusionMatrix: List[List[int]
                                            ] = self.__createConfusionMatrix(
                                                self.__protectedGroup
                                            )
        unprotectedGroupConfusionMatrix: List[List[int]
                                              ] = self.__createConfusionMatrix(
                                                  self.__unprotectedGroup
                                              )

        # Calcuate True Positive Rate and False Positive Rate for both groups
        protectedGroupFalsePositiveRate: float = protectedGroupConfusionMatrix[
            0][1] / (
                protectedGroupConfusionMatrix[0][0] +
                protectedGroupConfusionMatrix[0][1]
            )
        unprotectedGroupFalsePositiveRate: float = unprotectedGroupConfusionMatrix[
            0][1] / (
                unprotectedGroupConfusionMatrix[0][0] +
                unprotectedGroupConfusionMatrix[0][1]
            )

        print("************EPredictive Equality************")
        print(
            'Protected Group False Positive Rate: ',
            protectedGroupFalsePositiveRate
        )
        print(
            'Unprotected Group False Positive Rate: ',
            unprotectedGroupFalsePositiveRate
        )
        print("***************************************")

        return protectedGroupFalsePositiveRate == unprotectedGroupFalsePositiveRate

    def checkConditionalUseAccuracyEquality(self) -> bool:
        # Calculate confusion matrix for protected and unprotected groups
        protectedGroupConfusionMatrix: List[List[int]
                                            ] = self.__createConfusionMatrix(
                                                self.__protectedGroup
                                            )
        unprotectedGroupConfusionMatrix: List[List[int]
                                              ] = self.__createConfusionMatrix(
                                                  self.__unprotectedGroup
                                              )

        # Calcuate Positive and Negative Predictive Values for both groups
        protectedGroupPPV: float = protectedGroupConfusionMatrix[1][1] / (
            protectedGroupConfusionMatrix[1][1] +
            protectedGroupConfusionMatrix[0][1]
        )
        unprotectedGroupPPV: float = unprotectedGroupConfusionMatrix[1][1] / (
            unprotectedGroupConfusionMatrix[1][1] +
            unprotectedGroupConfusionMatrix[0][1]
        )
        protectedGroupNPV: float = protectedGroupConfusionMatrix[0][0] / (
            protectedGroupConfusionMatrix[0][0] +
            protectedGroupConfusionMatrix[1][0]
        )
        unprotectedGroupNPV: float = unprotectedGroupConfusionMatrix[0][0] / (
            unprotectedGroupConfusionMatrix[0][0] +
            unprotectedGroupConfusionMatrix[1][0]
        )

        print("************Conditional Use Accuracy Equality************")
        print('Protected Group Positive Predictive Value: ', protectedGroupPPV)
        print('Protected Group Negative Predictive Value: ', protectedGroupNPV)
        print(
            'Unprotected Group Positive Predictive Value: ',
            unprotectedGroupPPV
        )
        print(
            'Unprotected Group Negative Predictive Value: ',
            unprotectedGroupNPV
        )
        print("*********************************************************")

        return protectedGroupPPV == unprotectedGroupPPV and protectedGroupNPV == unprotectedGroupNPV

    def checkPredictiveParity(self) -> bool:
        # Calculate confusion matrix for protected and unprotected groups
        protectedGroupConfusionMatrix: List[List[int]
                                            ] = self.__createConfusionMatrix(
                                                self.__protectedGroup
                                            )
        unprotectedGroupConfusionMatrix: List[List[int]
                                              ] = self.__createConfusionMatrix(
                                                  self.__unprotectedGroup
                                              )

        # Calcuate Positive and Negative Predictive Values for both groups
        protectedGroupPPV: float = protectedGroupConfusionMatrix[1][1] / (
            protectedGroupConfusionMatrix[1][1] +
            protectedGroupConfusionMatrix[0][1]
        )
        unprotectedGroupPPV: float = unprotectedGroupConfusionMatrix[1][1] / (
            unprotectedGroupConfusionMatrix[1][1] +
            unprotectedGroupConfusionMatrix[0][1]
        )

        print("************Predictive Parity****************************")
        print('Protected Group Positive Predictive Value: ', protectedGroupPPV)
        print(
            'Unprotected Group Positive Predictive Value: ',
            unprotectedGroupPPV
        )
        print("*********************************************************")

        return protectedGroupPPV == unprotectedGroupPPV

    def checkOverallAccuracyEquality(self) -> bool:
        # Calculate confusion matrix for protected and unprotected groups
        protectedGroupConfusionMatrix: List[List[int]
                                            ] = self.__createConfusionMatrix(
                                                self.__protectedGroup
                                            )
        unprotectedGroupConfusionMatrix: List[List[int]
                                              ] = self.__createConfusionMatrix(
                                                  self.__unprotectedGroup
                                              )

        # Calcuate Overall Accuracy for both groups
        protectedGroupOverallAccuracy: float = (
            protectedGroupConfusionMatrix[0][0] +
            protectedGroupConfusionMatrix[1][1]
        ) / (
            protectedGroupConfusionMatrix[0][0] +
            protectedGroupConfusionMatrix[0][1] +
            protectedGroupConfusionMatrix[1][0] +
            protectedGroupConfusionMatrix[1][1]
        )
        unprotectedGroupOverallAccuracy: float = (
            unprotectedGroupConfusionMatrix[0][0] +
            unprotectedGroupConfusionMatrix[1][1]
        ) / (
            unprotectedGroupConfusionMatrix[0][0] +
            unprotectedGroupConfusionMatrix[0][1] +
            unprotectedGroupConfusionMatrix[1][0] +
            unprotectedGroupConfusionMatrix[1][1]
        )

        print("************Overall Accuracy Equality************")
        print(
            'Protected Group Overall Accuracy: ', protectedGroupOverallAccuracy
        )
        print(
            'Unprotected Group Overall Accuracy: ',
            unprotectedGroupOverallAccuracy
        )
        print("************************************************")

        return protectedGroupOverallAccuracy == unprotectedGroupOverallAccuracy
