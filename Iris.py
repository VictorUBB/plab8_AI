from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import neural_network
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn import linear_model

from MyNeuralNetwork import MyNeuralNetwork


class IrisProblem:

    def loadIrisData(self):
        from sklearn.datasets import load_iris

        data = load_iris()
        inputs = data['data']
        outputs = data['target']
        outputNames = data['target_names']
        featureNames = list(data['feature_names'])
        feature1 = [feat[featureNames.index('sepal length (cm)')] for feat in inputs]
        feature2 = [feat[featureNames.index('petal length (cm)')] for feat in inputs]
        inputs = [[feat[featureNames.index('sepal length (cm)')], feat[featureNames.index('petal length (cm)')]] for
                  feat in inputs]
        return inputs, outputs, outputNames

    def splitData(self,inputs, outputs):
        np.random.seed(5)
        indexes = [i for i in range(len(inputs))]
        trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
        testSample = [i for i in indexes if not i in trainSample]

        trainInputs = [inputs[i] for i in trainSample]
        trainOutputs = [outputs[i] for i in trainSample]
        testInputs = [inputs[i] for i in testSample]
        testOutputs = [outputs[i] for i in testSample]

        return trainInputs, trainOutputs, testInputs, testOutputs

    def normalisation(self,trainData, testData):
        scaler = StandardScaler()
        if not isinstance(trainData[0], list):
            # encode each sample into a list
            trainData = [[d] for d in trainData]
            testData = [[d] for d in testData]

            scaler.fit(trainData)  # fit only on training data
            normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
            normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

            # decode from list to raw values
            normalisedTrainData = [el[0] for el in normalisedTrainData]
            normalisedTestData = [el[0] for el in normalisedTestData]
        else:
            scaler.fit(trainData)  # fit only on training data
            normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
            normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
        return normalisedTrainData, normalisedTestData

    def data2FeaturesMoreClasses(self,inputs, outputs):
        labels = set(outputs)
        noData = len(inputs)
        for crtLabel in labels:
            x = [inputs[i][0] for i in range(noData) if outputs[i] == crtLabel]
            y = [inputs[i][1] for i in range(noData) if outputs[i] == crtLabel]
            # plt.scatter(x, y, label=outputNames[crtLabel])
        plt.xlabel('feat1')
        plt.ylabel('feat2')
        plt.legend()
        plt.show()

    def training(self,classifier, trainInputs, trainOutputs):
        # step3: training the classifier
        # identify (by training) the classification model
        classifier.fit(trainInputs, trainOutputs)

    def classification(self,classifier, testInputs):
        # step4: testing (predict the labels for new inputs)
        # makes predictions for test data
        computedTestOutputs = classifier.predict(testInputs)

        return computedTestOutputs

    def evalMultiClass(self,realLabels, computedLabels, labelNames):
        from sklearn.metrics import confusion_matrix

        confMatrix = confusion_matrix(realLabels, computedLabels)
        acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
        precision = {}
        recall = {}
        for i in range(len(labelNames)):
            precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
            recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
        return acc, precision, recall, confMatrix

    def plotConfusionMatrix(self,cm, classNames, title):
        from sklearn.metrics import confusion_matrix
        import itertools

        classes = classNames
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix ' + title)
        plt.colorbar()
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames, rotation=45)
        plt.yticks(tick_marks, classNames)

        text_format = 'd'
        thresh = cm.max() / 2.
        for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(column, row, format(cm[row, column], text_format),
                     horizontalalignment='center',
                     color='white' if cm[row, column] > thresh else 'black')

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        plt.show()
    def run(self):
        inData, outData, featNames = self.loadIrisData()
        print(featNames)
        print(inData[0], inData[50], inData[-5])
        print(outData[0], outData[50], outData[-5])
        inputs, outputs, outputNames =self.loadIrisData()
        trainInputs, trainOutputs, testInputs, testOutputs = self.splitData(inputs, outputs)

        # plot the training data distribution on classes
        plt.hist(trainOutputs, 3, rwidth=0.8)
        plt.xticks(np.arange(len(outputNames)), outputNames)
        plt.show()

        # plot the data in order to observe the shape of the classifier required in this problem
        self.data2FeaturesMoreClasses(trainInputs, trainOutputs)

        # normalise the data
        trainInputs, testInputs = self.normalisation(trainInputs, testInputs)

        classifier = neural_network.MLPClassifier()

        classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=100,
                                                  solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)

        self.training(classifier, trainInputs, trainOutputs)

        predictedLabels = self.classification(classifier, testInputs)

        acc, prec, recall, cm = self.evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)
        self.plotConfusionMatrix(cm, outputNames, "iris classification")
        myClassifier= MyNeuralNetwork(hiddenLayers=10)
        myClassifier.fit(np.array(trainInputs),np.array(trainOutputs))
        predLabels=classifier.predict(testInputs)
        acc1, prec1, recall1, cm1 = self.evalMultiClass(np.array(testOutputs), np.array(predLabels), outputNames)
        self.plotConfusionMatrix(cm1,outputNames,"my classification")
        print('acc: ', acc)
        print('precision: ', prec)
        print('recall: ', recall)

    def loadDigitData(self):
        from sklearn.datasets import load_digits

        data = load_digits()
        inputs = data.images
        outputs = data['target']
        outputNames = data['target_names']

        # shuffle the original data
        noData = len(inputs)
        permutation = np.random.permutation(noData)
        inputs = inputs[permutation]
        outputs = outputs[permutation]

        return inputs, outputs, outputNames

    def flatten(self,mat):
        x = []
        for line in mat:
            for el in line:
                x.append(el)
        return x

    def runDigits(self):
        inputs, outputs, outputNames = self.loadDigitData()
        trainInputs, trainOutputs, testInputs, testOutputs = self.splitData(inputs, outputs)
        # check if the data is uniform distributed over classes
        plt.hist(trainOutputs, rwidth=0.8)
        plt.xticks(np.arange(len(outputNames)), outputNames)
        plt.show()

        trainInputsFlatten = [self.flatten(el) for el in trainInputs]
        testInputsFlatten = [self.flatten(el) for el in testInputs]
        trainInputsNormalised, testInputsNormalised = self.normalisation(trainInputsFlatten, testInputsFlatten)
        # try to play by MLP parameters (e.g. change the HL size from 10 to 20 and see how this modification impacts the accuracy)
        classifier = neural_network.MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=100,
                                                  solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)

        self.training(classifier, trainInputsNormalised, trainOutputs)
        predictedLabels = self.classification(classifier, testInputsNormalised)
        acc, prec, recall, cm = self.evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)



        self.plotConfusionMatrix(cm, outputNames, "digit classification")
        print('acc: ', acc)
        print('precision: ', prec)
        print('recall: ', recall)

        # plot first 50 test images and their real and computed labels
        n = 10
        m = 5
        fig, axes = plt.subplots(n, m, figsize=(7, 7))
        fig.tight_layout()
        for i in range(0, n):
            for j in range(0, m):
                axes[i][j].imshow(testInputs[m * i + j])
                if (testOutputs[m * i + j] == predictedLabels[m * i + j]):
                    font = 'normal'
                else:
                    font = 'bold'
                axes[i][j].set_title(
                    'real ' + str(testOutputs[m * i + j]) + '\npredicted ' + str(predictedLabels[m * i + j]),
                    fontweight=font)
                axes[i][j].set_axis_off()

        plt.show()