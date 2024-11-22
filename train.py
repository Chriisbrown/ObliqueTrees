from dataset import DataSet
from XGBmodel import XGBoostClassifierModel
from YDFmodel import YDFClassifierModel,YDFObliqueModel
import matplotlib.pyplot as plt
import ydf


DataSet = DataSet.fromOpenML('hls4ml_lhc_jets_hlf')
DataSet.generate_test_train()
DataSet.save_test_train_h5('hls4ml_lhc_jets_hlf/')

ydfmodel = YDFClassifierModel("ydfmodel","YDF/")
ydfmodel.load_data_fromTrainTest('hls4ml_lhc_jets_hlf/')
ydfmodel.train()
ydfmodel.save()
ydfmodel.load()
ydfmodel.test()
ydfmodel.evaluate()