from model import ClassifierModel
import xgboost as xgb
from pathlib import Path
import os
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class XGBoostClassifierModel(ClassifierModel):
    def __init__(self,name,filepath):
        super().__init__(name,filepath)

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
        self.model = self.model.fit(self.DataSet.X_train[self.training_features], self.DataSet.y_train,verbose=10)

    def test(self):
        self.y_predict = self.model.predict_proba(self.DataSet.X_test[self.training_features].to_numpy())

    def save(self):
        self.model.save_model(self.filepath + self.name + ".json")

    def load(self):
        self.model.load_model(self.filepath + self.name +".json")

    def tune(self):

        # Define the hyperparameter grid
        param_dist = {
                    'learning_rate': stats.uniform(0.01, 0.2),
                    'subsample': stats.uniform(0, 0.5),
                    'min_child_weight' : stats.uniform(0,10),
                    'gamma' : stats.uniform(0,1),
                    'reg_lambda' : stats.uniform(0,1),
                    'reg_alpha' : stats.uniform(0,1)
                    }

        grid_search = RandomizedSearchCV(self.model, param_dist, cv=5, scoring='accuracy',n_jobs=-1,refit=True,n_iter=20,verbose=2)

        # Fit the GridSearchCV object to the training data
        grid_search.fit(self.DataSet.X_train[self.training_features], self.DataSet.y_train)
        self.model = grid_search.best_estimator_
        self.test()

        # Print the best set of hyperparameters and the corresponding score
        
        f = open(self.filepath+"hyperparameterscan.txt", "w")
        f.write("Best set of hyperparameters: "+ str(grid_search.best_params_)+'\n')
        f.write("Best score: "+str(accuracy_score(self.DataSet.y_test, np.argmax(self.y_predict,axis=1))))
        f.close()

    def conifer(self):
        import conifer
        from scipy.special import softmax
        # Create a conifer config
        cfg = conifer.backends.xilinxhls.auto_config()
        # Set the output directory to something unique
        cfg['OutputDir'] = 'prj_'+self.name

        # Create and compile the model
        # We need to pass the Booster object to conifer, so from xgboost's scikit-learn API,
        # we call bst.get_booster()
        self.conifer_model = conifer.converters.convert_from_xgboost(self.model, cfg)
        #self.conifer_model.draw(filename=self.filepath+self.name+".png")
        #plt.figure(figsize=(9, 9))
        #self.conifer_model.profile()
        #plt.savefig(self.filepath + "profile.png")
        print("num leaves: ",self.conifer_model.n_leaves())
        print("num nodes: ",self.conifer_model.n_nodes())
        print("sparsity: ",self.conifer_model.sparsity())

        #self.conifer_model.compile()

        # Run HLS C Simulation and get the output
        # xgboost 'predict' returns a probability like sklearn 'predict_proba'
        # so we need to compute the probability from the decision_function returned
        # by the HLS C Simulation
        #y_hls = softmax(self.conifer_model.decision_function(self.DataSet.X_test), axis=1)
        #y_xgb = self.model.predict_proba(self.DataSet.X_test)
