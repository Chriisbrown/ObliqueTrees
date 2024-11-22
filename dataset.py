from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import pandas as pd
import gc
import datetime
import json


class DataSet:
    def __init__(self,  name):

        self.name = name

        self.X = None
        self.y = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.train = None
        self.test = None

        self.verbose = 1

        self.random_state = 4
        self.test_size = 0.1

        self.num_classes = 5

        self.labelencoder = None

        self.training_features = []

        self.config_dict = {"name": self.name,
                            "datatransformed": False,
                            "databitted": False,
                            "datanormalised": False,
                            "testandtrain": False,
                            "testandtrainfilepath": "",
                            "numEvents": 0,
                            "numTrainEvents":0,
                            "numTestEvents":0,
                            "trainingFeatures": None,
                            "randomState": self.random_state,
                            "testsize": self.test_size,
                            "save_timestamp": None,
                            "loaded_timestamp": None
                            }
        
    @classmethod
    def fromOpenML(cls, datasetname):
        openmlclass = cls("open ML dataset: "+datasetname)
        openmlclass.load_data_from_OpenML(datasetname=datasetname)
        return openmlclass

    @classmethod
    def fromTrainTest(cls, filepath):
        traintest = cls("From Train Test")
        traintest.load_test_train_h5(filepath=filepath)
        traintest.training_features = traintest.config_dict["trainingFeatures"]
        return traintest
    
    @classmethod
    def fromSklearn(cls,loadfunction):
        sklearn = cls("sklearn dataset: ")
        sklearn.load_data_from_sklearn(loadfunction)
        return sklearn
        

    def load_data_from_sklearn(self,function):
        data = function()
        self.X, self.y = data.data, data.target
        self.config_dict["trainingFeatures"] = data['feature_names']
        self.training_features = data['feature_names']

        self.X = pd.DataFrame(self.X)
        self.y = pd.DataFrame(self.y)


    def load_data_from_OpenML(self,datasetname):
        data = fetch_openml(datasetname)
        self.X, self.y = data['data'], data['target']
        self.config_dict["trainingFeatures"] = data['feature_names']
        self.training_features = data['feature_names']

    def generate_test_train(self):
        self.labelencoder = LabelEncoder()
        y = self.labelencoder.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=self.test_size, random_state=self.random_state)
        self.y_train  = pd.DataFrame({'target': self.y_train },dtype=int)
        self.y_test = pd.DataFrame({'target': self.y_test},dtype=int)

        self.y_train.dropna(inplace=True)
        self.X_train.dropna(inplace=True)
        self.y_test.dropna(inplace=True)
        self.X_test.dropna(inplace=True)

        self.config_dict["numEvents"] = self.X.shape[0]
        self.config_dict["numTrainEvents"] = self.X_train.shape[0]
        self.config_dict["numTestEvents"] = self.X_test.shape[0]
        self.config_dict["testandtrain"] = True


    def transform_data(self):
        pass

    def bit_data(self, normalise: bool = False):
        pass

    def save_test_train_h5(self, filepath):

        Path(filepath).mkdir(parents=True, exist_ok=True)

        X_train_store = pd.HDFStore(filepath+'X_train.h5')
        X_test_store = pd.HDFStore(filepath+'X_test.h5')
        y_train_store = pd.HDFStore(filepath+'y_train.h5')
        y_test_store = pd.HDFStore(filepath+'y_test.h5')

        X_train_store['df'] = self.X_train  # save it
        X_test_store['df'] = self.X_test  # save it
        y_train_store['df'] = self.y_train  # save it
        y_test_store['df'] = self.y_test  # save it

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        X_train_store.close()
        X_test_store.close()
        y_train_store.close()
        y_test_store.close()

        self.config_dict["testandtrainfilepath"] = filepath
        self.config_dict["save_timestamp"] = datetime.datetime.now().strftime(
            "%H:%M %d/%m/%y")
        with open(filepath+'config_dict.json', 'w') as f:
            json.dump(self.config_dict, f, indent=4)

        if self.verbose == 1:
            print("===Train Test Saved====")

    def load_test_train_h5(self, filepath):
        X_train_file = Path(filepath+'X_train.h5')
        if X_train_file.is_file():
            X_train_store = pd.HDFStore(filepath+'X_train.h5')
            self.X_train = X_train_store['df']
            X_train_store.close()
        else:
            print("No X Train h5 File")

        X_test_file = Path(filepath+'X_test.h5')
        if X_test_file .is_file():
            X_test_store = pd.HDFStore(filepath+'X_test.h5')
            self.X_test = X_test_store['df']
            X_test_store.close()
        else:
            print("No X Test h5 File")

        y_train_file = Path(filepath+'y_train.h5')
        if y_train_file.is_file():
            y_train_store = pd.HDFStore(filepath+'y_train.h5')
            self.y_train = y_train_store['df']
            y_train_store.close()
        else:
            print("No y train h5 file")

        y_test_file = Path(filepath+'y_test.h5')
        if y_test_file.is_file():
            y_test_store = pd.HDFStore(filepath+'y_test.h5')
            self.y_test = y_test_store['df']
            y_test_store.close()
        else:
            print("No y test h5 file")

        config_dict_file = Path(filepath+'config_dict.json')
        if config_dict_file.is_file():
            with open(filepath+'config_dict.json', 'r') as f:
                self.config_dict = json.load(f)
            self.config_dict["loaded_timestamp"] = datetime.datetime.now().strftime(
                "%H:%M %d/%m/%y")
            self.name = self.config_dict["name"]
        else:
            print("No configuration dictionary json file")

    def write_hls_file(self, filepath, num_events=10):
        # self.transform_data()
        # self.bit_data()
        Path(filepath).mkdir(parents=True, exist_ok=True)
        with open(filepath+"input_hls.txt", 'w') as f:
            for i in range(num_events):
                for j,feat in enumerate(self.config_dict["trainingFeatures"]):
                    f.write("{j}".format(int(self.X.iloc[i][feat])))
                f.write("\n")
        f.close()