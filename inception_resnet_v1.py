# Implementation of Inception-ResNet-v1 architecture
# Author: Shobhit Lamba
# e-mail: slamba4@uic.edu

# Importing the libraries
from keras.layers import Input, add
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def resnet_stem(input):
    '''The stem of the Inception-ResNet-v1 network.'''
    
    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = Conv2D(32, (3, 3), activation = "relu", strides = (2, 2), padding = "same")(input) # 149 * 149 * 32
    x = Conv2D(32, (3, 3), activation = "relu", padding = "same")(x) # 147 * 147 * 32
    x = Conv2D(64, (3, 3), activation = "relu", padding = "same")(x) # 147 * 147 * 64
    
    x = MaxPooling2D((3, 3), strides = (2, 2))(x) # 73 * 73 * 64
    
    x = Conv2D(80, (1, 1), activation = "relu", padding = "same")(x) # 73 * 73 * 80
    x = Conv2D(192, (3, 3), activation = "relu", padding = "same")(x) # 71 * 71 * 192
    x = Conv2D(256, (3, 3), activation = "relu", strides = (2, 2), padding = "same")(x) # 35 * 35 * 256
    
    x = BatchNormalization(axis = -1)(x)
    x = Activation("relu")(x)
    
    return x

def inception_resnet_A(input, scale_residual = True):
    '''Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.'''
        
    ar1 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)
    
    ar2 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)
    ar2 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ar2)
    
    ar3 = Conv2D(32, (1, 1), activation = "relu", padding = "same")(input)
    ar3 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ar3)
    ar3 = Conv2D(32, (3, 3), activation = "relu", padding = "same")(ar3)
    
    merged = concatenate([ar1, ar2, ar3], axis = -1)
    
    ar = Conv2D(256, (1, 1), activation = "linear", padding = "same")(merged)
    if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)
    
    output = add([input, ar])
    output = BatchNormalization(axis = -1)(output)
    output = Activation("relu")(output)
    
    return output

def inception_resnet_B(input, scale_residual = True):
    '''Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.'''
    
    br1 = Conv2D(128, (1, 1), activation = "relu", padding = "same")(input)
    
    br2 = Conv2D(128, (1, 1), activation = "relu", padding = "same")(input)
    br2 = Conv2D(128, (1, 7), activation = "relu", padding = "same")(br2)
    br2 = Conv2D(128, (7, 1), activation = "relu", padding = "same")(br2)
    
    merged = concatenate([br1, br2], axis = -1)
    
    br = Conv2D(896, (1, 1), activation = "linear", padding = "same")(merged)
    if scale_residual: br = Lambda(lambda b: b * 0.1)(br)
    
    output = add([input, br])
    output = BatchNormalization(axis = -1)(output)
    output = Activation("relu")(output)
    
    return output

def inception_resnet_C(input, scale_residual = True):
    '''Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.'''
    
    cr1 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)
    
    cr2 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)
    cr2 = Conv2D(192, (1, 3), activation = "relu", padding = "same")(cr2)
    cr2 = Conv2D(192, (3, 1), activation = "relu", padding = "same")(cr2)
    
    merged = concatenate([cr1, cr2], axis = -1)
    
    cr = Conv2D(1792, (1, 1), activation = "linear", padding = "same")(merged)
    if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)
    
    output = add([input, cr])
    output = BatchNormalization(axis = -1)(output)
    output = Activation("relu")(output)
    
    return output

def reduction_resnet_A(input):
    '''Architecture of a 35 * 35 to 17 * 17 Reduction_ResNet_A block.'''
    
    rar1 = MaxPooling2D((3,3), strides = (2,2))(input)

    rar2 = Conv2D(384, (3, 3), activation = "relu", strides = (2,2))(input)

    rar3 = Conv2D(192, (1, 1), activation = "relu", padding = "same")(input)
    rar3 = Conv2D(224, (3, 3), activation = "relu", padding = "same")(rar3)
    rar3 = Conv2D(256, (3, 3), activation = "relu", strides = (2,2))(rar3)

    merged = concatenate([rar1, rar2, rar3], axis = -1)
    rar = BatchNormalization(axis = -1)(merged)
    rar = Activation("relu")(rar)
    
    return rar

def reduction_resnet_B(input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.'''
    
    rbr1 = MaxPooling2D((3,3), strides = (2,2), padding = "valid")(input)
    
    rbr2 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
    rbr2 = Conv2D(384, (3, 3), activation = "relu", strides = (2,2))(rbr2)
    
    rbr3 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
    rbr3 = Conv2D(256, (3, 3), activation = "relu", strides = (2,2))(rbr3)
    
    rbr4 = Conv2D(256, (1, 1), activation = "relu", padding = "same")(input)
    rbr4 = Conv2D(256, (3, 3), activation = "relu", padding = "same")(rbr4)
    rbr4 = Conv2D(256, (3, 3), activation = "relu", strides = (2,2))(rbr4)
    
    merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis = -1)
    rbr = BatchNormalization(axis = -1)(merged)
    rbr = Activation("relu")(rbr)
    
    return rbr

def inception_resnet_v1(nb_classes = 1001, scale = True):
    '''Creates the Inception_ResNet_v1 network.'''
    
    init = Input((299, 299, 3)) # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering
    
    # Input shape is 299 * 299 * 3
    x = resnet_stem(init) # Output: 35 * 35 * 256
    
     # 5 x Inception A
    for i in range(5):
        x = inception_resnet_A(x, scale_residual = scale)
        # Output: 35 * 35 * 256
        
    # Reduction A
    x = reduction_resnet_A(x) # Output: 17 * 17 * 896

    # 10 x Inception B
    for i in range(10):
        x = inception_resnet_B(x, scale_residual = scale)
        # Output: 17 * 17 * 896
        
    # Reduction B
    x = reduction_resnet_B(x) # Output: 8 * 8 * 1792

    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_C(x, scale_residual = scale) 
        # Output: 8 * 8 * 1792
        
    # Average Pooling
    x = AveragePooling2D((8, 8))(x) # Output: 1792

    # Dropout
    x = Dropout(0.2)(x) # Keep dropout 0.2 as mentioned in the paper
    x = Flatten()(x) # Output: 1792

    # Output layer
    output = Dense(units = nb_classes, activation = "softmax")(x) # Output: 10000

    model = Model(init, output, name = "Inception-ResNet-v1")   
        
    return model


if __name__ == "__main__":
    inception_resnet_v1 = inception_resnet_v1()
    inception_resnet_v1.summary()
    