from model import ClassifierModel
import ydf
from pathlib import Path
import os
from dataset import DataSet
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class YDFClassifierModel(ClassifierModel):
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

    def tune(self):
        tuner = ydf.RandomSearchTuner(num_trials=24)

        
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

    def load(self):
        self.model = ydf.load_model(self.filepath + self.name)

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
        self.conifer_model = conifer.converters.convert_from_ydf(self.model, cfg)
        # self.conifer_model.draw(filename=self.filepath+self.name+".png")
        # plt.figure(figsize=(9, 9))
        # self.conifer_model.profile()
        # plt.savefig(self.filepath + "profile.png")
        with open('out.txt', 'a') as f:
            print(str(self.conifer_model.n_leaves()) + "," + str(self.conifer_model.n_nodes()) + "," + str(self.conifer_model.sparsity()) + ",", file=f, end='')
        #print("num leaves: ",self.conifer_model.n_leaves())
        #print("num nodes: ",self.conifer_model.n_nodes())
        #print("sparsity: ",self.conifer_model.sparsity())

    def describe(self):

        importances = self.model.variable_importances()['SUM_SCORE']
        print(importances)
        labels = [importances[i][0] for i in range(len(importances))]
        importances = [importances[i][1] for i in range(len(importances))]

        print(labels)
        print(importances)

        #self.model.describe()
        #self.model.print_tree()
        tree = self.model.get_tree(tree_idx=1)
        print(tree.root.value)

        print(tree.pretty(self.model.data_spec()))



class YDFObliqueModel(YDFClassifierModel):
    def __init__(self,name,filepath):
        super().__init__(name,filepath)

        self.model = ydf.GradientBoostedTreesLearner(   num_trees=100,
                                                        max_depth=3,
                                                        apply_link_function=False,                                
                                                        shrinkage=0.1,
                                                        subsample=1.0,
                                                        split_axis="SPARSE_OBLIQUE",
                                                        sparse_oblique_normalization="MIN_MAX",
                                                        sparse_oblique_num_projections_exponent=0.8,
                                                        sparse_oblique_projection_density_factor=1.0,
                                                        sparse_oblique_weights='BINARY',
                                                        num_candidate_attributes_ratio=0.5,
                                                        min_examples=7,
                                                        early_stopping="MIN_LOSS_FINAL",
                                                        growing_strategy="LOCAL",
                                                        #early_stopping='LOSS_INCREASE',
                                                        #max_num_nodes=3,
                                                        use_hessian_gain = True,
                                                        label=self.label
                                                    )

    def train(self):
        self.model = self.model.train(self.DataSet.train)

    def tune(self):
        tuner = ydf.RandomSearchTuner(num_trials=24)

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

    def describe(self):
        importances = self.model.variable_importances()['SUM_SCORE']
        print(importances)
        labels = [importances[i][0] for i in range(len(importances))]
        importances = [importances[i][1] for i in range(len(importances))]

        print(labels)
        print(importances)
    
        self.model.describe()
        self.model.print_tree()
        tree = self.model.get_tree(tree_idx=0)
        print(tree)

        print(tree.pretty(self.model.data_spec()))


