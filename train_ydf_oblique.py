from dataset import DataSet
from model import YDFObliqueModel
import matplotlib.pyplot as plt
import ydf

# Generate the dataset from the hls4ml_lhc_jets_hlf openML dataset
# Generate test and train and save them in the unique directory hls4ml_lhc_jets_hlf/
# Only needed once, once the dataset is pulled and saved in the directory it can be opened by the model
DataSet = DataSet.fromOpenML('hls4ml_lhc_jets_hlf')
DataSet.generate_test_train()
DataSet.save_test_train_h5('hls4ml_lhc_jets_hlf/')

# Create the model and give it a unique name and saving directory, plots and model files will be saved here
ydf_oblique_model = YDFObliqueModel("ydf_oblique_model","YDF_oblique_test/")
# Load the previously saved datset
ydf_oblique_model.load_data_fromTrainTest('hls4ml_lhc_jets_hlf/')
# Train and save to the directory
ydf_oblique_model.train()
ydf_oblique_model.save()
# Load the model (not necessary here but as an example) using the unique name and directory
ydf_oblique_model.load("ydf_oblique_model","YDF_oblique_test/")
# Predict on the test dataset
ydf_oblique_model.test()
# Synthesize the model
ydf_oblique_model.synth_model()
# evaluate on the test and generate a ROC plot
ydf_oblique_model.evaluate()

# Hyperparameter scan of the model across the X_train dataset, trials can be set depending on how big a scan is needed
ydf_oblique_model.tune(n_trials=1)
