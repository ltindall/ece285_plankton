'''
Uses keras-vis library for Dense layer visualizations of 12 plankton categories
'''

from matplotlib import pyplot as plt

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_activation

from keras.models import load_model
from keras.layers import Activation
from scipy.misc import imsave

# Load the fine-tuned plankton model
model = load_model('plankton_model.h5')
# Change output activation from softmax to linear
model.layers[-1].activation = Activation('linear')
print('Model loaded.')

# The name of the layer we want to visualize
layer_name = 'dense_1'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Get dense layer visualizations for all 12 plankton categories
vis_images = []
for idx in range(12):
    img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=1000 )
    img = utils.draw_text(img, str(idx))
    vis_images.append(img)

stitched = utils.stitch_images(vis_images)   
imsave('class_%d.png' % (idx), stitched)

plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()
