import keras
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn
from keras.callbacks import History

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
    
    tl_model = getModel( output_dim ) 
    
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

    tl_model.save('plankton_model.h5')
    
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
    
    
    
