'''
Performs gradient ascent to generate image that maximizes a particular filter within VGG
https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
'''

from keras import applications
from keras import backend as K
import numpy as np
from scipy.misc import imsave
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x,0,1)

    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x

# Load VGG16
model = applications.VGG16(include_top=False,weights='imagenet')

model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers][1:])

layer_name = 'block1_conv2'

input_width = 299
input_height = 299
input_img = model.input

#%%
# Visualizes first 200 filters in layer_name
for filter_index  in range(0,200):
    print('Filter ' + str(filter_index))
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:,:,:,filter_index])

    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([input_img], [loss, grads])

    step = 1

    input_img_data = (np.random.random((1,input_width, input_height, 3))- 0.5) * 20 + 128.

    # Gradient ascent for 50 iterations
    for i in range(50):
        print(i)
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value*step
        if loss_value <= 0.:
            break

  
        if loss_value > 0.:
            img = deprocess_image(input_img_data[0])
            imsave('%s_filter_%d.png' % (layer_name, filter_index), img)