from sklearn import metrics
import numpy as np
import joblib
import pickle
from pathlib import Path
import os
from dataset import DataSet
import plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical



class ClassifierModel:
    def __init__(self,name="",filepath="./"):

        self.name = name
        self.filepath = filepath
        self.DataSet = None
        
        self.model = None
        self.y_predict = None
        self.y_predict_proba = None

        self.training_features = None
        self.target_labels = ['g','q','t','w','z']

        if not Path(filepath).is_dir():
            os.system("mkdir -p " + filepath)

    def load_data(self,DataSet):
        self.DataSet = DataSet
        self.training_features = DataSet.training_features

    def load_data_fromOpenML(self,openmlname,testtrain = True):
        self.DataSet = DataSet.fromOpenML(openmlname)
        self.training_features = self.DataSet.training_features
        if testtrain:
            self.DataSet.generate_test_train()

    def load_data_fromTrainTest(self,directory):
        self.DataSet = DataSet.fromTrainTest(directory)
        
        self.training_features = self.DataSet.training_features

        self.DataSet.X_train['target'] = self.DataSet.y_train.to_numpy()
        self.DataSet.train = self.DataSet.X_train

        self.DataSet.X_test['target'] = self.DataSet.y_test.to_numpy()
        self.DataSet.test = self.DataSet.X_test



    def train(self):
        pass

    def test(self):
        pass

    def evaluate(self):
        print("Accuracy: {}".format(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))))
        with open('out.txt', 'a') as f:
            print(str(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))) + ",", file=f, end='')
        # y_test_cat = to_categorical(self.DataSet.test['target'], self.DataSet.num_classes)
        # plt.figure(figsize=(9, 9))
        
        # _ = plotting.makeRoc(y_test_cat, self.y_predict,self.target_labels, name=self.name )
        # plt.savefig(self.filepath + "test.png")

    def save(self,name):
        pass

    def load(self,name):
        pass

    def synth_model(self):
        pass