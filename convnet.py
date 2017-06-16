import keras
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn
from keras.callbacks import History
from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import merge
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Lambda
import theano.tensor as T

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original Alexnet
    combing the conventkeras and pylearn functions.
    erralves
    """
    def f(X):

        ch, r, c, b = X.shape
        half = n // 2
        sq = T.sqr(X)

        extra_channels = T.alloc(0., ch + 2*half, r, c, b)
        sq = T.set_subtensor(extra_channels[half:half+ch,:,:,:], sq)

        scale = k
        for i in range(n):
            scale += alpha * sq[i:i+ch,:,:,:]

        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output = X[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = X[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = X[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = X[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def g(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f, output_shape=lambda input_shape: g(input_shape), **kwargs)

def AlexNet(output_dim, weights_path=None, heatmap=False):
    if heatmap:
        inputs = Input(shape=(3, None, None))
    else:
        inputs = Input(shape=(3, 224, 224))

    conv_1 = Convolution2D(96, 11, 11, subsample=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096, 6, 6, activation='relu', name='dense_1')(dense_1)
        dense_2 = Convolution2D(4096, 1, 1, activation='relu', name='dense_2')(dense_1)
        dense_3 = Convolution2D(1000, 1, 1, name='dense_3')(dense_2)
        prediction = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(output_dim, name='dense_3')(dense_3)
        prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(input=inputs, output=prediction)

    model.compile(optimizer='adam', 
           loss='categorical_crossentropy',
           metrics=['accuracy'])
    model.summary()

    if weights_path:
        model.load_weights(weights_path)

    return model

def getModel( output_dim ):
    ''' 
        * output_dim: the number of classes (int)
        
        * return: compiled model (keras.engine.training.Model)
    '''
    vgg_model = VGG16( weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[-2].output #Last FC layer's output  
    
    #Create softmax layer taking input as vgg_out
    softmax_layer = keras.layers.core.Dense(output_dim,
                          init='lecun_uniform',
                          activation='softmax')(vgg_out)
                          
    #Create new transfer learning model
    tl_model = Model( input=vgg_model.input, output=softmax_layer )
    
    #Freeze all layers of VGG16 and Compile the model
    for layers in vgg_model.layers:
        layers.trainable = False;
    '''    
    tl_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    '''
    tl_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
    #Confirm the model is appropriate
    tl_model.summary()

    return tl_model

if __name__ == '__main__':
    #Output dim for your dataset
    output_dim = 12 #For Caltech97
    
    #tl_model = getModel( output_dim ) 

    tl_model = AlexNet(output_dim)
    
    # Input data generator
    train_datagen = ImageDataGenerator(
        samplewise_center = True)
        
    train_generator = train_datagen.flow_from_directory(
        'keras_images/train/',
        target_size=(224, 224),
        batch_size=50,
        class_mode='categorical')
    
    test_datagen = ImageDataGenerator(
        samplewise_center = True)
        
    test_generator = test_datagen.flow_from_directory(
        'keras_images/test/',
        target_size=(224, 224),
        batch_size=50,
        class_mode='categorical')
    
    #Train the model
    history = tl_model.fit_generator(train_generator,
        steps_per_epoch = 50,
        epochs = 200,
        verbose = 1,
        validation_data = test_generator,
        validation_steps = 20)

    tl_model.save('plankton_model_alexnet.h5')
    
    #Test the model
    plt.plot(history.history['acc'])
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.show()
    
    plt.figure()
    plt.plot(history.history['val_acc'])
    plt.show()
    
    plt.figure()
    plt.plot(history.history['val_loss'])
    plt.show()
    
    
    
