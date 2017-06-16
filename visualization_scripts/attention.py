'''
Generates the attention maps for a particular plankton category using grad-CAM
Based off of keras-vis library
'''

import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model, Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency, visualize_cam

from os import listdir

from scipy.misc import imsave

#%%
# Load plankton trained VGG model
model = load_model('plankton_model.h5')
print('Model loaded.')

#%%
# The name of the layer we want to visualize
layer_name = 'dense_1'
#layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Use 10 images corresponding to plankton category
plankton = 'acanthera'

category = 'image_data/'+ plankton +'_files/'
image_paths = listdir(category)
image_paths = image_paths[:10]

heatmaps = []
for path in image_paths:
    seed_img = utils.load_img(category + path, target_size=(224, 224))
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    
    # Samplewise Centering
    x = x - np.mean(x, axis=3, keepdims=True)
    pred_class = np.argmax(model.predict(x))
    
    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img, alpha=0.5)
    heatmaps.append(heatmap)
    
    imsave('%s.png' % (category + '_' + path), heatmap)
    
plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title(plankton)
plt.show()
