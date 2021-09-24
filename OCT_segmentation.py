import os
import model as mt
from keras.optimizers import *
import pandas as pd
from utils import Params, Logging, TrainOps, Evaluation
from python_generator import DataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import cv2



def figure1(data, model_dir):
    '''
    data is a list of lists containing train validation and test results for Spectralies and Topcon OCTs seperately
    [result_dict["train_spectralis"].tolist(),
    result_dict["train_topcon"].tolist(),
    result_dict["validation_spectralis"].tolist(),
    result_dict["validation_topcon"].tolist(),
    result_dict["test_spectralis"].tolist(),
    result_dict["test_topcon"].tolist()]
    '''

    plt.figure(figsize = (30, 10))
    fig7, ax7 = plt.subplots(figsize = (20, 10))

    bplot_ = ax7.boxplot(data, showmeans = False, patch_artist = True)

    colors = ["black", "darkred", "black", "darkred", "black", "darkred"]
    for patch, color in zip(bplot_['boxes'], colors):
        patch.set_facecolor(color)
    plt.setp(bplot_["medians"], color = "gold", linewidth = 3)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tick_params(
        axis = 'x',  # changes apply to the x-axis
        which = 'both',  # both major and minor ticks are affected
        bottom = False,  # ticks along the bottom edge are off
        top = False,  # ticks along the top edge are off
        labelbottom = False)

    # fill with colors
    # fill with colors
    plt.savefig(model_dir + "/segmentation_results.png", dpi = 450, transparent = True)


# load utils classes
params = Params("params.json")
params.data_dir = "./data"
params.model_directory = "./output"
params.continuing_training = False
params.batchnorm = True
params.is_training = True
params.shuffle = True
params.mode = "test"

# instantiate eval class
evalutaion = Evaluation(params)

# get train ops
trainops = TrainOps(params)


# get model
model = mt.get_deep_unet(params)
model.summary()

'''train and save model'''
save_model_path = os.path.join(params.model_directory, "weights.hdf5")
'''Load models trained weights'''
model.load_weights(save_model_path, by_name = True, skip_mismatch = True)

# store predictions
if not os.path.exists(os.path.join(params.model_directory, params.mode + "_predictions")):
    os.makedirs(os.path.join(params.model_directory, params.mode + "_predictions"))


record_id = 'test_noise'
im = cv2.imread(os.path.join('./', record_id + '.jpg'))
# convert to three channel
if im.shape[-1] != 3:
    im = np.stack((im,) * 3, axis = -1)
im_resized = cv2.resize(im, (params.img_shape, params.img_shape)).reshape(params.img_shape, params.img_shape, 3)
test_im = np.divide(im_resized, 255., dtype=np.float32)
test_im = np.nan_to_num(test_im)
predicted_mask = model.predict(test_im.reshape(1, params.img_shape, params.img_shape, 3))
prediction = predicted_mask[0, :, :, 0]
# set prediction to classes
prediction[prediction < 0.5] = 0
prediction[prediction >= 0.5] = 1
prediction = prediction*255
cv2.imwrite('mask_noise.jpg', prediction.astype(int))

           