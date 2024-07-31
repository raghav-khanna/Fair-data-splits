from types import FunctionType
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

classifierMapper: dict[str, FunctionType] = {'KNN': KNeighborsClassifier, 'SVC': SVC, 'GNB': GaussianNB, 'DeT': DecisionTreeClassifier}


class ClassifierClass:

    def __init__(self, trainSet: [], testSet: [], classifierName: str = 'KNN', ) -> None:
        self.__classifierName: str = classifierName
        self.__xTrain: [] = [vals[0:len(vals) - 1] for vals in trainSet]
        self.__xTest: [] = [vals[0:len(vals) - 1] for vals in testSet]
        self.__yTrain: [] = [vals[-1] for vals in trainSet]
        self.__yTest: [] = [vals[-1] for vals in testSet]
        self.__yPredict: [] = []
        self.__confusionMatrix: [[]] = []
        print('To be trained on: ', self.__xTrain, 'with targets: ', self.__yTrain)
        print('To be tested on: ', self.__xTest, 'with targets: ', self.__yTest)

    def evaluate(self, **kwargs):
        parameterisedClassifier = []
        if classifierMapper[self.__classifierName] == KNeighborsClassifier:
            k = int(kwargs.get('k'))
            parameterisedClassifier = Pipeline(steps = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors = k))])
        elif classifierMapper[self.__classifierName] == SVC:
            # To check for relevant config options
            parameterisedClassifier = SVC()
        elif classifierMapper[self.__classifierName] == GaussianNB:
            # To check for relevant config options
            parameterisedClassifier = GaussianNB()
        elif classifierMapper[self.__classifierName] == DecisionTreeClassifier:
            # To check for relevant config options
            parameterisedClassifier = DecisionTreeClassifier()
        parameterisedClassifier.fit(self.__xTrain, self.__yTrain)
        self.__yPredict = parameterisedClassifier.predict(self.__xTest)
        self.__confusionMatrix = confusion_matrix(self.__yTest, self.__yPredict)
        return self.__confusionMatrix

    def performance(self):
        print('**********\nPerformance of', self.__classifierName, ': ')
        print('\nTrained on:')
        print('|Attributes|Target|')
        for i in range(len(self.__xTrain)):
            print('|', self.__xTrain[i], '  |  ', self.__yTrain[i], '  |')
        print('\n\nTest Results:')
        print('|Attributes|Expected|Predicted|')
        for i in range(len(self.__xTest)):
            print('|', self.__xTest[i], '  |  ', self.__yTest[i], '  |  ', self.__yPredict[i], '|')
        print('\n\nConfusion Matrix:')
        print('| True Negative:', self.__confusionMatrix[0][0], '|False Positive:', self.__confusionMatrix[0][1], '|')
        print('|False Negative:', self.__confusionMatrix[1][0], '| True Positive:', self.__confusionMatrix[1][1], '|')
        print('\n\n**********')
