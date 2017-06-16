from keras import applications
from keras import backend as K
import numpy as np
from scipy.misc import imsave
from keras.models import Sequential, Model, load_model
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
	

#model = applications.VGG16(include_top=True,weights='imagenet')
#model = applications.resnet50.ResNet50(include_top=False, weights = 'imagenet')
#model = applications.inception_v3.InceptionV3(include_top=True, weights='imagenet')
model = load_model('plankton_model.h5')

model.summary()

input_width = 224
input_height = 224
input_img = model.input

#%%
for category in range(0,1):
	print('Category ' + str(category))
	loss = model.output[:,category]

	grads = K.gradients(loss, input_img)[0]
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	iterate = K.function([input_img], [loss, grads])

	step = 1

	input_img_data = (np.random.random((1,input_width, input_height, 3))- 0.5) * 20 + 128.

	for i in range(1000):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value*step
		print('Loss: ' + str(loss_value))
		'''
		if loss_value <= 0. or loss_value > .98:
			break
		'''
	img = deprocess_image(input_img_data[0])
	imsave('class_%d.png' % (category), img)
		
		
