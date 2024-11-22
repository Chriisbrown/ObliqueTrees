from dataset import DataSet
from XGBmodel import XGBoostClassifierModel
from YDFmodel import YDFClassifierModel,YDFObliqueModel
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import ydf
import time



ydfmodel = YDFClassifierModel("ydfmodel","YDF/")
ydfmodel.load_data_fromTrainTest('hls4ml_lhc_jets_hlf/')
ydfmodel.train()
ydfmodel.tune()
ydfmodel.save()
ydfmodel.load()
ydfmodel.test()
ydfmodel.evaluate()
     
# ydfobqmodel = YDFObliqueModel("ydfobqmodel","YDFobq/")
# ydfobqmodel.load_data_fromTrainTest('hls4ml_lhc_jets_hlf/')
# ydfobqmodel.train()
# ydfobqmodel.tune()
# ydfobqmodel.save()
# ydfobqmodel.load()
# ydfobqmodel.test()
# ydfobqmodel.evaluate()