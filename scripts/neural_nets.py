import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Dense, Dropout, Flatten
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import MaxPool2D, GlobalAveragePooling2D







#################################
# model classes 
#################################
class All_Models():

    def __init__(self, height, width, bins, years):
        """
        the image array params
        """
        self.height = height
        self.width  = width
        self.bins   = bins
        self.years  = years
    

    def multi_input_cf_net_Ext(self):

        inputE3 = Input(shape=(int(self.height), int(self.width), self.bins))
        inputE2 = Input(shape=(int(self.height), int(self.width), self.bins))
        inputE1 = Input(shape=(int(self.height), int(self.width), self.bins))
        inputE0 = Input(shape=(int(self.height), int(self.width), self.bins))
    

        A0 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputE3)
        A0_2 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001))(A0) 
        A0_M = Model(inputs=inputE3, outputs=A0_2)

        A = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputE2)
        A_2 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001))(A)
        A_M = Model(inputs=inputE2, outputs=A_2)
  
        B0 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputE1)
        B0_3 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001))(B0)
        B0_M = Model(inputs=inputE1, outputs=B0_3)
  
  
        C = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputE0)
        C_2 = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001))(C)
        C_2M = Model(inputs=inputE0, outputs=C_2)
  
  
        # concatenate
        ab_conc = Concatenate(axis=-1)([A0_M.output, A_M.output, B0_M.output, C_2M.output])

        ### after concatenate
        net = BatchNormalization()(ab_conc)
        net = Conv2D(64, (3,3), activation='relu', padding="same")(net)
        net = MaxPooling2D()(net)
        net = Conv2D(64, (3,3), activation='relu', padding="same")(net)
        net = MaxPooling2D()(net)
        # net = Dropout(0.3)(net)
        net = Flatten()(net)
        net = Dense(32, activation='relu')(net)
        #net = Dropout(0.4)(net)
        net = Dense(16, activation='relu')(net)
        net = Dropout(0.5)(net)
        #net = Dense(16, activation='relu')(net)
        outputs = Dense(3, activation='softmax')(net)
        model = Model(inputs=[A0_M.input, A_M.input, B0_M.input, C_2M.input], outputs=[outputs], name='2DCNN-CTAEXT')

        return model
