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
import ydf
import xgboost as xgb
import scipy.stats as stats
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import conifer



class ClassifierModel:
    '''
    Classifier model template
    '''
    def __init__(self,name="",filepath="./"):

        self.name = name
        self.filepath = filepath
        self.DataSet = None
        
        self.model = None
        self.y_predict = None
        self.y_predict_proba = None

        self.training_features = None
        self.target_labels = None

        if not Path(filepath).is_dir():
            os.system("mkdir -p " + filepath)

    def load_data(self,DataSet):
        '''
        Load data from a dataset class
        '''
        self.DataSet = DataSet
        self.training_features = DataSet.training_features
        self.target_labels = DataSet.target_labels

    def load_data_fromOpenML(self,openmlname,testtrain = True):
        '''
        Directly load data from openML creating a dataset class and saving the data as h5
        '''
        self.DataSet = DataSet.fromOpenML(openmlname)
        self.training_features = self.DataSet.training_features
        self.target_labels = self.DataSet.target_labels
        if testtrain:
            self.DataSet.generate_test_train()

    def load_data_fromTrainTest(self,directory):
        '''
        Load data directly from a saved version of the dataset in h5 format
        '''
        self.DataSet = DataSet.fromTrainTest(directory)
        
        self.training_features = self.DataSet.training_features
        self.target_labels = self.DataSet.target_labels

        self.DataSet.X_train['target'] = self.DataSet.y_train.to_numpy()
        self.DataSet.train = self.DataSet.X_train

        self.DataSet.X_test['target'] = self.DataSet.y_test.to_numpy()
        self.DataSet.test = self.DataSet.X_test

    def train(self):
        pass

    def test(self):
        pass

    def evaluate(self):
        '''
        Evaluate model, using accuracy metric
        Plot the reciever operating characteristic curve
        '''
        print("Accuracy: {}".format(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))))
        categories = self.DataSet.test['target'].astype('category').to_numpy()
        y_test_cat = np.zeros((len(self.DataSet.test['target']),max(categories)+1))
        for i in range(len(categories)):
            y_test_cat[i][categories[i]] = 1
        print(y_test_cat)
        plt.figure(figsize=(9, 9))
        
        _ = plotting.makeRoc(y_test_cat, self.y_predict,self.target_labels, name=self.name )
        plt.savefig(self.filepath + "roc.png")
        
    def evaluate_synth(self):
        print(self.DataSet.X_test.to_numpy())
        self.y_cpp_predict = self.cpp_model.decision_function(self.DataSet.X_test.to_numpy())
        self.y_hls_predict = self.hls_model.decision_function(self.DataSet.X_test.to_numpy())
        
        print("YDF Accuracy: {}".format(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))))
        print("CPP Accuracy: {}".format(accuracy_score(self.DataSet.y_test, np.argmax(self.y_cpp_predict,axis=1))))
        print("HLS Accuracy: {}".format(accuracy_score(self.DataSet.y_test, np.argmax(self.y_hls_predict ,axis=1))))
        
        
        categories = self.DataSet.test['target'].astype('category').to_numpy()
        y_test_cat = np.zeros((len(self.DataSet.test['target']),max(categories)+1))
        for i in range(len(categories)):
            y_test_cat[i][categories[i]] = 1
        print(y_test_cat)
        plt.figure(figsize=(9, 9))
        
        _ = plotting.makeRoc(y_test_cat, self.y_cpp_predict,self.target_labels, name=self.name )
        plt.savefig(self.filepath + "cpp_roc.png")
        

    def save(self,name):
        pass

    def load(self,name):
        pass

    def synth_model(self):
        cpp_cfg = conifer.backends.cpp.auto_config()
        # cpp_cfg["Precision"] = "float"  # Optional float precision.
        cpp_cfg["OutputDir"] = "prj_cpp"
        #cpp_cfg['score_precision'] = 'ap_fixed<18,8>'
        #cpp_cfg['threshold_precision'] = f"ap_fixed<{input_width},{input_integer}>"
        #cpp_cfg["input_precision"] = f"ap_fixed<{input_width},{input_integer}>"
        #cpp_cfg["weight_precision"] = weight_type
        # # Convert the YDF model to a C++ Conifer model.
        self.cpp_model = conifer.converters.convert_from_ydf(self.model, cpp_cfg)
        self.cpp_model.compile()


        # Create a conifer config
        hls_cfg = conifer.backends.xilinxhls.auto_config()
        # hls_cfg["Precision"] = "float"  # Optional float precision.
        hls_cfg["OutputDir"] = "prj_hls"
        #hls_cfg['score_precision'] = 'ap_fixed<18,8>'
        #hls_cfg['threshold_precision'] = f"ap_fixed<{input_width},{input_integer}>"
        #hls_cfg["input_precision"] = f"ap_fixed<{input_width},{input_integer}>"
        #hls_cfg["weight_precision"] = weight_type

        # Convert the YDF model to a HLS Conifer model.
        self.hls_model = conifer.converters.convert_from_ydf(self.model, hls_cfg)
        self.hls_model.compile()


        self.hls_model.build(vsynth=True)

        report = self.hls_model.read_report()

        print(report)


class XGBoostClassifierModel(ClassifierModel):
    '''
    XGBoost classifier model based on Classifier Template
    '''
    def __init__(self,name,filepath):
        super().__init__(name,filepath)

        # Generic hyperparameters, to be changed for a specific task or found via tuning
        self.model = xgb.XGBClassifier(n_estimators=100, 
                                       max_depth=3,
                                       learning_rate=0.1986,
                                       subsample=0.1822,
                                       gamma=0.428,
                                       min_child_weight=8.6637,
                                       reg_alpha= 0.1290,
                                       reg_lambda=0.8852,
                                       objective='multi:softmax')
                                       
    def train(self):
        self.model = self.model.fit(self.DataSet.X_train[self.training_features], self.DataSet.y_train,verbose=1)

    def test(self):
        self.y_predict = self.model.predict_proba(self.DataSet.X_test[self.training_features].to_numpy())

    def save(self):
        self.model.save_model(self.filepath + self.name + ".json")

    def load(self):
        self.model.load_model(self.filepath + self.name +".json")

    def tune(self,ntrials=24):
        '''
        Tune a model based on a random grid search cross validation targetting standard hyperparameters for xgboost 
        '''

        # Define the hyperparameter grid
        param_dist = {
                    'learning_rate': stats.uniform(0.01, 0.2),
                    'subsample': stats.uniform(0, 0.5),
                    'min_child_weight' : stats.uniform(0,10),
                    'gamma' : stats.uniform(0,1),
                    'reg_lambda' : stats.uniform(0,1),
                    'reg_alpha' : stats.uniform(0,1)
                    }

        grid_search = RandomizedSearchCV(self.model, param_dist, cv=5, scoring='accuracy',n_jobs=-1,refit=True,n_iter=n_trials,verbose=2)

        # Fit the GridSearchCV object to the training data
        grid_search.fit(self.DataSet.X_train[self.training_features], self.DataSet.y_train)
        self.model = grid_search.best_estimator_
        self.test()

        # Print the best set of hyperparameters and the corresponding score
        f = open(self.filepath+"hyperparameterscan.txt", "w")
        f.write("Best set of hyperparameters: "+ str(grid_search.best_params_)+'\n')
        f.write("Best score: "+str(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))))
        f.close()

class YDFClassifierModel(ClassifierModel):
    '''
    Yggdrasil classifier model based on Classifier Template
    '''
    def __init__(self,name,filepath):
        super().__init__(name,filepath)

        self.label = "target"

        self.model = ydf.GradientBoostedTreesLearner(   num_trees=10,
                                                        max_depth=3,
                                                        apply_link_function=False,                               
                                                        shrinkage=0.01,
                                                        split_axis="AXIS_ALIGNED",
                                                        num_candidate_attributes_ratio=0.8,
                                                        min_examples=10,
                                                        early_stopping="MIN_LOSS_FINAL",
                                                        growing_strategy="BEST_FIRST_GLOBAL",
                                                        use_hessian_gain = True,
                                                        label=self.label
                                                    )

    def train(self):
        self.model = self.model.train(self.DataSet.train)

    def tune(self,n_trails=24):
        tuner = ydf.RandomSearchTuner(num_trials=n_trails)
        
        tuner.choice("shrinkage", [0.02,0.05,0.1])
        tuner.choice("subsample", [0.6,0.8,0.9,1.0])
        tuner.choice("use_hessian_gain", [False,True])
        tuner.choice("min_examples",[5,7,10,20])
        tuner.choice("split_axis",["AXIS_ALIGNED"])
        tuner.choice("num_candidate_attributes_ratio",[0.2,0.5,0.9,1.0])

        local_subspace = tuner.choice("growing_strategy", ["LOCAL"])
        local_subspace.choice("max_depth", [5])

        global_subspace = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
        global_subspace.choice("max_num_nodes", [32])

        learner = ydf.GradientBoostedTreesLearner(
                    label="target",
                    num_trees=100, # Used for all the trials.
                    tuner=tuner,
                    apply_link_function = False
                    )
        self.model =learner.train(self.DataSet.train)

        logs = self.model.hyperparameter_optimizer_logs()
        top_score = max(t.score for t in logs.trials)
        selected_trial = [t for t in logs.trials if t.score == top_score][0] 
        self.test()
        
        f = open(self.filepath+"hyperparameterscan.txt", "w")
        f.write("Best set of hyperparameters: " + str(selected_trial.params)+'\n')
        f.write("Best score: "+str(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))))
        f.close()


    def test(self):
        self.y_predict = self.model.predict(self.DataSet.test)

    def save(self,):
        self.model.save(self.filepath + self.name)

    def load(self,name,filepath):
        self.model = ydf.load_model(filepath + name)

    # Create a model from a saved model
    @classmethod
    def fromSavedModel(cls,name,filepath):
        loadedmodel = cls(name,filepath)
        loadedmodel.model = ydf.load_model(filepath + name)
        return loadedmodel

class YDFObliqueModel(YDFClassifierModel):
    '''
    Yggdrasil oblique classifier model based on YDF classifier Template
    '''
    def __init__(self,name,filepath):
        super().__init__(name,filepath)

        self.model = ydf.GradientBoostedTreesLearner(   num_trees=100,
                                                        max_depth=3,
                                                        apply_link_function=False,                                
                                                        shrinkage=0.1,
                                                        subsample=0.6,
                                                        split_axis="SPARSE_OBLIQUE",
                                                        sparse_oblique_normalization="MIN_MAX",
                                                        sparse_oblique_num_projections_exponent=2.0,
                                                        sparse_oblique_projection_density_factor=4.0,
                                                        sparse_oblique_weights='CONTINUOUS',
                                                        num_candidate_attributes_ratio=1.0,
                                                        min_examples=20,
                                                        early_stopping="MIN_LOSS_FINAL",
                                                        growing_strategy="LOCAL",
                                                        use_hessian_gain = False,
                                                        label=self.label
                                                    )

    def train(self):
        self.model = self.model.train(self.DataSet.train)

    def tune(self,n_trials=24):
        tuner = ydf.RandomSearchTuner(num_trials=n_trials)

        tuner.choice("shrinkage", [0.1,0.5,1.0])
        tuner.choice("subsample", [0.6,0.8,0.9,1.0])
        tuner.choice("use_hessian_gain", [False,True])
        tuner.choice("min_examples",[5,7,10,20])
        tuner.choice("num_candidate_attributes_ratio",[0.2,0.5,0.9,1.0])
        tuner.choice("sparse_oblique_num_projections_exponent",[0.8,1.0,1.2,2.0])
        tuner.choice("sparse_oblique_projection_density_factor",[1.0,2.0,3.0,4.0,5.0])
        tuner.choice("sparse_oblique_normalization",["NONE","STANDARD_DEVIATION","MIN_MAX"])
        tuner.choice("sparse_oblique_weights",["CONTINUOUS"])
        tuner.choice("split_axis",["SPARSE_OBLIQUE"])

        local_subspace = tuner.choice("growing_strategy", ["LOCAL"])
        local_subspace.choice("max_depth", [5])

        global_subspace = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
        global_subspace.choice("max_num_nodes", [32])

        learner = ydf.GradientBoostedTreesLearner(
                    label="target",
                    num_trees=100, # Used for all the trials.
                    apply_link_function=False,
                    tuner=tuner,
                    )
        self.model =learner.train(self.DataSet.train)

        logs = self.model.hyperparameter_optimizer_logs()
        top_score = max(t.score for t in logs.trials)
        selected_trial = [t for t in logs.trials if t.score == top_score][0] 
        self.test()
        
        f = open(self.filepath+"hyperparameterscan.txt", "w")
        f.write("Best set of hyperparameters: " + str(selected_trial.params)+'\n')
        f.write("Best score: " + str(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))))
        f.close()

