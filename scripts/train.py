##############################
# only difference from check_classif_Ext_CTA.py
# is on the plotting routine for conv layer activation
# we select specific name for conv layer
# we select only one layer output 
##############################

import time
import matplotlib.pyplot as plt
import classification_DataLoader_CTA_Ext, neural_nets
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

##############################
# libraries for NN
##############################
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import categorical_accuracy

start_time = time.time()

########################################
# read npy files-- C0
########################################

print ("!!!!!!!!!!!!! started reading source data !!!!!!!!!!!!!")
print ("!!!!!!!! source class 0, energy bin 0")
im_arr_C0_E0, C0_E0_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=0, bin=0, less_data=True)
print ('num ims: ', len(im_arr_C0_E0), im_arr_C0_E0.shape)

###################################
print ("!!!!!!!! source class 0, energy bin 1")
im_arr_C0_E1, C0_E1_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=0, bin=1, less_data=True)
print ('num ims: ', len(im_arr_C0_E1), im_arr_C0_E1.shape)

###################################

print ("!!!!!!!! source class 0, energy bin 2")
im_arr_C0_E2, C0_E2_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=0, bin=2, less_data=True)
print ('num ims: ', len(im_arr_C0_E2), im_arr_C0_E2.shape)

###################################

print ("!!!!!!!! source class 0, energy bin 3")
im_arr_C0_E3, C0_E3_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=0, bin=3, less_data=True)
print ('num ims: ', len(im_arr_C0_E3), im_arr_C0_E3.shape)


############################
# -- C1
###########################
print ("!!!!!!!! source class 1, energy bin 0")
im_arr_C1_E0, C1_E0_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=1, bin=0, less_data=True)
print ('num ims: ', len(im_arr_C1_E0), im_arr_C1_E0.shape)

print ("!!!!!!!! source class 1, energy bin 1")
im_arr_C1_E1, C1_E1_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=1, bin=1, less_data=True)
print ('num ims: ', len(im_arr_C1_E1), im_arr_C1_E1.shape)


print ("!!!!!!!! source class 1, energy bin 2")
im_arr_C1_E2, C1_E2_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=1, bin=2, less_data=True)
print ('num ims: ', len(im_arr_C1_E2), im_arr_C1_E2.shape)


print ("!!!!!!!! source class 1, energy bin 3")
im_arr_C1_E3, C1_E3_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=1, bin=3, less_data=True)
print ('num ims: ', len(im_arr_C1_E3), im_arr_C1_E3.shape)

###################################


############################
# --- C2
###########################
print ("!!!!!!!! source class 2, energy bin 0")
im_arr_C2_E0, C2_E0_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=2, bin=0, less_data=True)
print ('num ims: ', len(im_arr_C2_E0), im_arr_C2_E0.shape)

print ("!!!!!!!! source class 2, energy bin 1")
im_arr_C2_E1, C2_E1_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=2, bin=1, less_data=True)
print ('num ims: ', len(im_arr_C2_E1), im_arr_C2_E1.shape)


print ("!!!!!!!! source class 2, energy bin 2")
im_arr_C2_E2, C2_E2_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=2, bin=2, less_data=True)
print ('num ims: ', len(im_arr_C2_E2), im_arr_C2_E2.shape)


print ("!!!!!!!! source class 2, energy bin 3")
im_arr_C2_E3, C2_E3_fnames = classification_DataLoader_CTA_Ext.read_train_source_ext(cls_label=2, bin=3, less_data=True)
print ('num ims: ', len(im_arr_C2_E3), im_arr_C2_E3.shape)

###################################


##########################################################
# separate into train and test before augmentation
##########################################################

frac_train = 0.85 # [percentage of data used for training]


im_arr_C0_E0_train, fnames_C0_E0_train = im_arr_C0_E0[:int(len(im_arr_C0_E0)*frac_train)], C0_E0_fnames[:int(len(C0_E0_fnames)*frac_train)]
im_arr_C0_E1_train, fnames_C0_E1_train = im_arr_C0_E1[:int(len(im_arr_C0_E1)*frac_train)], C0_E1_fnames[:int(len(C0_E1_fnames)*frac_train)]
im_arr_C0_E2_train, fnames_C0_E2_train = im_arr_C0_E2[:int(len(im_arr_C0_E2)*frac_train)], C0_E2_fnames[:int(len(C0_E0_fnames)*frac_train)]
im_arr_C0_E3_train, fnames_C0_E3_train = im_arr_C0_E3[:int(len(im_arr_C0_E3)*frac_train)], C0_E3_fnames[:int(len(C0_E0_fnames)*frac_train)]


print ('check lens: ', len(im_arr_C0_E0_train), len(im_arr_C0_E3_train))


im_arr_C1_E0_train, fnames_C1_E0_train = im_arr_C1_E0[:int(len(im_arr_C1_E0)*frac_train)], C1_E0_fnames[:int(len(C1_E0_fnames)*frac_train)]
im_arr_C1_E1_train, fnames_C1_E1_train = im_arr_C1_E1[:int(len(im_arr_C1_E1)*frac_train)], C1_E1_fnames[:int(len(C1_E1_fnames)*frac_train)]
im_arr_C1_E2_train, fnames_C1_E2_train = im_arr_C1_E2[:int(len(im_arr_C1_E2)*frac_train)], C1_E2_fnames[:int(len(C1_E2_fnames)*frac_train)]
im_arr_C1_E3_train, fnames_C1_E3_train = im_arr_C1_E3[:int(len(im_arr_C1_E3)*frac_train)], C1_E3_fnames[:int(len(C1_E3_fnames)*frac_train)]

print ('check lens: ', len(im_arr_C1_E0_train), len(im_arr_C1_E3_train))


im_arr_C2_E0_train, fnames_C2_E3_train = im_arr_C2_E0[:int(len(im_arr_C2_E0)*frac_train)], C2_E0_fnames[:int(len(C0_E0_fnames)*frac_train)]
im_arr_C2_E1_train, fnames_C2_E3_train = im_arr_C2_E1[:int(len(im_arr_C2_E1)*frac_train)], C2_E1_fnames[:int(len(C0_E0_fnames)*frac_train)]
im_arr_C2_E2_train, fnames_C2_E3_train = im_arr_C2_E2[:int(len(im_arr_C2_E2)*frac_train)], C2_E2_fnames[:int(len(C0_E0_fnames)*frac_train)]
im_arr_C2_E3_train, fnames_C2_E3_train = im_arr_C2_E3[:int(len(im_arr_C2_E3)*frac_train)], C2_E3_fnames[:int(len(C0_E0_fnames)*frac_train)]

print ('check lens: ', len(im_arr_C2_E0_train), len(im_arr_C2_E3_train))


### test

im_arr_C0_E0_test, fnames_C0_E0_test = im_arr_C0_E0[int(len(im_arr_C0_E0)*frac_train):], C0_E0_fnames[int(len(C0_E0_fnames)*frac_train):]
im_arr_C0_E1_test, fnames_C0_E1_test = im_arr_C0_E1[int(len(im_arr_C0_E1)*frac_train):], C0_E1_fnames[int(len(C0_E1_fnames)*frac_train):]
im_arr_C0_E2_test, fnames_C0_E2_test = im_arr_C0_E2[int(len(im_arr_C0_E2)*frac_train):], C0_E2_fnames[int(len(C0_E2_fnames)*frac_train):]
im_arr_C0_E3_test, fnames_C0_E3_test = im_arr_C0_E3[int(len(im_arr_C0_E3)*frac_train):], C0_E3_fnames[int(len(C0_E3_fnames)*frac_train):]


print ('check lens test C0: ', len(im_arr_C0_E0_test), len(im_arr_C0_E3_test))

im_arr_C1_E0_test, fnames_C1_E0_test = im_arr_C1_E0[int(len(im_arr_C1_E0)*frac_train):], C1_E0_fnames[int(len(C1_E0_fnames)*frac_train):]
im_arr_C1_E1_test, fnames_C1_E1_test = im_arr_C1_E1[int(len(im_arr_C1_E1)*frac_train):], C1_E1_fnames[int(len(C1_E1_fnames)*frac_train):]
im_arr_C1_E2_test, fnames_C1_E2_test = im_arr_C1_E2[int(len(im_arr_C1_E2)*frac_train):], C1_E2_fnames[int(len(C1_E2_fnames)*frac_train):]
im_arr_C1_E3_test, fnames_C1_E3_test = im_arr_C1_E3[int(len(im_arr_C1_E3)*frac_train):], C1_E3_fnames[int(len(C1_E3_fnames)*frac_train):]


print ('check lens test C1: ', len(im_arr_C1_E0_test), len(im_arr_C1_E3_test))

im_arr_C2_E0_test, fnames_C2_E0_test = im_arr_C2_E0[int(len(im_arr_C2_E0)*frac_train):], C2_E0_fnames[int(len(C2_E0_fnames)*frac_train):]
im_arr_C2_E1_test, fnames_C2_E1_test = im_arr_C2_E1[int(len(im_arr_C2_E1)*frac_train):], C2_E1_fnames[int(len(C2_E1_fnames)*frac_train):]
im_arr_C2_E2_test, fnames_C2_E2_test = im_arr_C2_E2[int(len(im_arr_C2_E2)*frac_train):], C2_E2_fnames[int(len(C2_E2_fnames)*frac_train):]
im_arr_C2_E3_test, fnames_C2_E3_test = im_arr_C2_E3[int(len(im_arr_C2_E3)*frac_train):], C2_E3_fnames[int(len(C2_E3_fnames)*frac_train):]


print ('check lens test C2: ', len(im_arr_C2_E0_test), len(im_arr_C2_E3_test))



##########################################
# preparing for augmentation
##########################################

### C0

im_C0_E0_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C0_E0_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C0, E0:', im_C0_E0_train_aug.shape)


im_C0_E1_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C0_E1_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C0, E1:', im_C0_E1_train_aug.shape)

im_C0_E2_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C0_E2_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C0, E2:', im_C0_E2_train_aug.shape)


im_C0_E3_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C0_E3_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C0, E3:', im_C0_E3_train_aug.shape)


### C1



im_C1_E0_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C1_E0_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C1, E0:', im_C1_E0_train_aug.shape)


im_C1_E1_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C1_E1_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C1, E1:', im_C1_E1_train_aug.shape)

im_C1_E2_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C1_E2_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C1, E2:', im_C1_E2_train_aug.shape)


im_C1_E3_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C1_E3_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C1, E3:', im_C1_E3_train_aug.shape)


### C2


im_C2_E0_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C2_E0_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C2, E0:', im_C2_E0_train_aug.shape)


im_C2_E1_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C2_E1_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C2, E1:', im_C2_E1_train_aug.shape)

im_C2_E2_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C2_E2_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C2, E2:', im_C2_E2_train_aug.shape)


im_C2_E3_train_aug = classification_DataLoader_CTA_Ext.aug_ims_source_CTA(im_arr_C2_E3_train, rotate_90=True, rotate_180=False, augment=True)


print ('check shapes after augmentation:  C2, E3:', im_C2_E3_train_aug.shape)





######################################
# create arrays for labels: 
# int arrays of 0, 1, 2
# should be the same len as train and test sets of images
# done after augmentation of images 
######################################


lab_0_train = [0]*len(im_C0_E0_train_aug)
lab_1_train = [1]*len(im_C1_E0_train_aug)
lab_2_train = [2]*len(im_C2_E0_train_aug)

lab_0_train_arr= np.array(lab_0_train)
lab_1_train_arr= np.array(lab_1_train)
lab_2_train_arr= np.array(lab_2_train)



lab_0_test = [0]*len(im_arr_C0_E0_test)
lab_1_test = [1]*len(im_arr_C1_E0_test)
lab_2_test = [2]*len(im_arr_C2_E0_test)

lab_0_test_arr= np.array(lab_0_test)
lab_1_test_arr= np.array(lab_1_test)
lab_2_test_arr= np.array(lab_2_test)



print ('check shapes of labels: ', lab_0_train_arr.shape, lab_1_train_arr.shape, lab_2_train_arr.shape)
print ('check shapes of labels: ', lab_0_test_arr.shape, lab_1_test_arr.shape, lab_2_test_arr.shape)




##############################################
# turn the labels into categorical
##############################################
# 3 classes

all_lab_cat_C0 = tf.keras.utils.to_categorical(lab_0_train_arr, num_classes=3, dtype='uint8')
lab_cat_C0_test = tf.keras.utils.to_categorical(lab_0_test_arr, num_classes=3, dtype='uint8')


print ('check consistency after turning labels to categorical C0:', '\n')

print ('check label and shapes: ', all_lab_cat_C0[2], lab_0_train_arr[2], all_lab_cat_C0.shape)



all_lab_cat_C1 = tf.keras.utils.to_categorical(lab_1_train_arr, num_classes=3, dtype='uint8')
lab_cat_C1_test = tf.keras.utils.to_categorical(lab_1_test_arr, num_classes=3, dtype='uint8')

print ('check consistency after turning labels to categorical C1:', '\n')

print ('check label and shapes: ', all_lab_cat_C1[2], lab_1_train_arr[2], all_lab_cat_C1.shape)


all_lab_cat_C2 = tf.keras.utils.to_categorical(lab_2_train_arr, num_classes=3, dtype='uint8')
lab_cat_C2_test = tf.keras.utils.to_categorical(lab_2_test_arr, num_classes=3, dtype='uint8')

print ('check consistency after turning labels to categorical C2:', '\n')

print ('check label and shapes: ', all_lab_cat_C2[2], lab_2_train_arr[2], all_lab_cat_C2.shape)


######################################
# concatenate different classes
######################################

C0_C1_C2_E0_ims_train = np.concatenate((im_C0_E0_train_aug, im_C1_E0_train_aug, im_C2_E0_train_aug))
print ('all ims concatenate E0: train part>', C0_C1_C2_E0_ims_train.shape)

C0_C1_C2_E1_ims_train = np.concatenate((im_C0_E1_train_aug, im_C1_E1_train_aug, im_C2_E1_train_aug))
print ('all ims concatenate E1: train part>', C0_C1_C2_E1_ims_train.shape)

C0_C1_C2_E2_ims_train = np.concatenate((im_C0_E2_train_aug, im_C1_E2_train_aug, im_C2_E2_train_aug))
print ('all ims concatenate E2: train part>', C0_C1_C2_E2_ims_train.shape)

C0_C1_C2_E3_ims_train = np.concatenate((im_C0_E3_train_aug, im_C1_E3_train_aug, im_C2_E3_train_aug))
print ('all ims concatenate E3: train part>', C0_C1_C2_E3_ims_train.shape)

## repeat the same for the test data

C0_C1_C2_E0_ims_test = np.concatenate((im_arr_C0_E0_test, im_arr_C1_E0_test, im_arr_C2_E0_test))
print ('all ims concatenate E0: test part>', C0_C1_C2_E0_ims_test.shape)

C0_C1_C2_E1_ims_test = np.concatenate((im_arr_C0_E1_test, im_arr_C1_E1_test, im_arr_C2_E1_test))
print ('all ims concatenate E1: test part>', C0_C1_C2_E1_ims_test.shape)

C0_C1_C2_E2_ims_test = np.concatenate((im_arr_C0_E2_test, im_arr_C1_E2_test, im_arr_C2_E2_test))
print ('all ims concatenate E2: train part>', C0_C1_C2_E2_ims_test.shape)

C0_C1_C2_E3_ims_test = np.concatenate((im_arr_C0_E3_test, im_arr_C1_E3_test, im_arr_C2_E3_test))
print ('all ims concatenate E3: train part>', C0_C1_C2_E3_ims_test.shape)

#################
# repeat same for labels
#################

C0_C1_C2_labs_cat = np.concatenate((all_lab_cat_C0, all_lab_cat_C1, all_lab_cat_C2))
print ('all labs concatenate: train part>', C0_C1_C2_labs_cat.shape)

C0_C1_C2_labs_cat_test = np.concatenate((lab_cat_C0_test, lab_cat_C1_test, lab_cat_C2_test))
print ('all labs concatenate: test part>', C0_C1_C2_labs_cat_test.shape)


C0_C1_C2_labs = np.concatenate((lab_0_train_arr, lab_1_train_arr, lab_2_train_arr))
print (C0_C1_C2_labs.shape)


#################################################
# shuffle and split the data
#################################################

E0_ims_shuffled, E1_ims_shuffled, E2_ims_shuffled, E3_ims_shuffled, all_labs_shuffled = shuffle(C0_C1_C2_E0_ims_train, C0_C1_C2_E1_ims_train,
                                                                                                C0_C1_C2_E2_ims_train, C0_C1_C2_E3_ims_train,
                                                                                                C0_C1_C2_labs_cat)

print ('check shuffle list shapes: ', '\n', E0_ims_shuffled.shape, E1_ims_shuffled.shape, E2_ims_shuffled.shape, all_labs_shuffled.shape)



E0_ims_train, E0_ims_valid, E1_ims_train, E1_ims_valid, E2_ims_train, E2_ims_valid, E3_ims_train, E3_ims_valid, all_labs_train, all_labs_valid = train_test_split(E0_ims_shuffled, 
                                                                                                                                                                  E1_ims_shuffled,
                                                                                                                                                                  E2_ims_shuffled,
                                                                                                                                                                  E3_ims_shuffled, all_labs_shuffled,
                                                                                                                                                                  test_size=0.20, random_state=20,)

print ('train_im E0 shape, train_im_E1_shape, train_im_E2_shape, test_im E3_shape, train_lab_shape, test_lab_shape: ', E0_ims_train.shape, E1_ims_train.shape, 
       E2_ims_train.shape, E3_ims_valid.shape, all_labs_train.shape, all_labs_valid.shape)

##########################################
# neural net 
##########################################

print ('\n')
print ('################################')
print ('#Neural Net Part')
print ('################################')



gen_model = neural_nets.All_Models(height=70, width=70, bins=1, years=10)
multi_input_cf_net_Ext_3class = gen_model.multi_input_cf_net_Ext()


# # # # # # # #########################
# # # # # # # #  define hyperparams
# # # # # # # # #######################    

print ('number of params in cnn model: ', multi_input_cf_net_Ext_3class.count_params())

print ('summary of Model')
print ('\n')

multi_input_cf_net_Ext_3class.summary()



epochs = 500
batch_size=32
learning_rate = 1e-4

es = EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta=1e-6)

red_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=8, min_lr=0.000000001)

homepath= "/path/to/dir/"
mcp_save = ModelCheckpoint(filepath=homepath+"Models/best_model_Ext_Source_weights_classf_C0_C1_C2_Normed_999.h5",
                           save_best_only=True, save_weights_only=True, 
                           monitor='val_loss', mode='min') # 999 signifies the training data has increased from 500 total images to 999 images


multi_input_cf_net_Ext_3class.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=[categorical_accuracy])



history = multi_input_cf_net_Ext_3class.fit(x=[E0_ims_train, E1_ims_train, E2_ims_train, E3_ims_train], y=all_labs_train, 
                                            validation_data=([E0_ims_valid, E1_ims_valid, E2_ims_valid, E3_ims_valid], all_labs_valid),
                                            batch_size=batch_size, epochs=epochs, callbacks=[es, red_lr, mcp_save])


end_time = time.time()

print ('total reading data + training time in minutes: ', (end_time-start_time)/60.)



#################################
# check the loss and acc.
#################################

fig = plt.figure(figsize=(7, 5))

train_loss_p = history.history['loss']
val_loss_p = history.history['val_loss']

train_cat_p = history.history['categorical_accuracy']
val_cat_p = history.history['val_categorical_accuracy']


fig.add_subplot(121)

plt.plot(range(len(train_loss_p)), train_loss_p, linestyle='-', color='r', label='Train Loss')
plt.plot(range(len(val_loss_p)), val_loss_p, linestyle='-.', color='b', label='Val Loss')

plt.xlabel('Num Epochs', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)

fig.add_subplot(122)

plt.plot(range(len(train_loss_p)), train_cat_p, linestyle='-', color='r', label='Train Cat. Acc')
plt.plot(range(len(val_loss_p)), val_cat_p, linestyle='-.', color='b', label='Val Cat. Acc')

plt.xlabel('Num Epochs', fontsize=13)
plt.ylabel('categorical Accuracy', fontsize=13)
plt.legend(fontsize=13)

plt.tight_layout()
plt.savefig(homepath + 'Images/plots/train_classif_3class_Ext1_999.png', dpi=200)

#####################################
# evaluate on validation data
#####################################

multi_input_cf_net_Ext_3class.load_weights(homepath + "Models/best_model_Ext_Source_weights_classf_C0_C1_C2_Normed_999.h5")

loss, cat_acc = multi_input_cf_net_Ext_3class.evaluate([E0_ims_valid, E1_ims_valid, E2_ims_valid, E3_ims_valid], all_labs_valid)                    

print ('loss and categorical accuracy validation: ', loss, cat_acc)
# # # # # # # # ###################################


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class_types = ['C0','C1', 'C2']

def conf_matrix(predictions, test_labels): 
    ''' Plots conf. matrix and classification report 
        test labels are assumed to be in categorical format
        predictions are from model.predict method'''

    test_labels_rounded = np.argmax(np.round(test_labels), axis=1)
    cm=confusion_matrix(test_labels_rounded, np.argmax(np.round(predictions, 3), axis=1))
    cf_matrix_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    print("Classification Report:\n")
    cr=classification_report(test_labels_rounded,
                                np.argmax(np.round(predictions), axis=1), 
                                target_names=[class_types[i] for i in range(len(class_types))])
    print(cr)
    plt.figure(figsize=(6, 6))
    sns_hmp = sns.heatmap(cf_matrix_norm, annot=True, xticklabels = [class_types[i] for i in range(len(class_types))], 
                yticklabels = [class_types[i] for i in range(len(class_types))], fmt="0.2f")
    plt.xlabel('Pred', fontsize = 12) # x-axis label with fontsize 15
    plt.ylabel('True', fontsize = 12)
    plt.suptitle(r'$0.03<C0(\sigma)<0.1; \, 0.1<C1(\sigma)<0.3;\, C2(\sigma)>0.3$', fontsize=11)            
    fig = sns_hmp.get_figure()
    fig.savefig(homepath + 'Images/CF_Matrix_3class_Extended_CTA_999.png', dpi=250)

###########################
# predict on test data
###########################

pred_class_model = multi_input_cf_net_Ext_3class.predict([C0_C1_C2_E0_ims_test, C0_C1_C2_E1_ims_test, C0_C1_C2_E2_ims_test, C0_C1_C2_E3_ims_test])

conf_matrix(pred_class_model, C0_C1_C2_labs_cat_test)    


#######################################
# plotting the conv layer activations
#######################################

layer_outputs = [layer.output for layer in multi_input_cf_net_Ext_3class.layers]
print ('check num layers: ', len(layer_outputs))
print ('check random layers: ', layer_outputs[6])
print ('check chosen layers: ', layer_outputs[14], layer_outputs[16])
activation_model = Model(inputs = multi_input_cf_net_Ext_3class.input, outputs=layer_outputs)

print ('num of test images and example shape: ', len(im_arr_C0_E0_test), im_arr_C0_E0_test[0].shape)


print ('check names: C0', fnames_C0_E0_test[2])
C0_test_E0_num1 = im_arr_C0_E0_test[2].reshape(1, 70, 70, 1)
C0_test_E1_num1 = im_arr_C0_E1_test[2].reshape(1, 70, 70, 1)
C0_test_E2_num1 = im_arr_C0_E2_test[2].reshape(1, 70, 70, 1)
C0_test_E3_num1 = im_arr_C0_E3_test[2].reshape(1, 70, 70, 1)

print ('check names: C1', fnames_C1_E0_test[5])
C1_test_E0_num1 = im_arr_C1_E0_test[5].reshape(1, 70, 70, 1)
C1_test_E1_num1 = im_arr_C1_E1_test[5].reshape(1, 70, 70, 1)
C1_test_E2_num1 = im_arr_C1_E2_test[5].reshape(1, 70, 70, 1)
C1_test_E3_num1 = im_arr_C1_E3_test[5].reshape(1, 70, 70, 1)

print ('check names: C2', fnames_C2_E0_test[5])
C2_test_E0_num1 = im_arr_C2_E0_test[5].reshape(1, 70, 70, 1)
C2_test_E1_num1 = im_arr_C2_E1_test[5].reshape(1, 70, 70, 1)
C2_test_E2_num1 = im_arr_C2_E2_test[5].reshape(1, 70, 70, 1)
C2_test_E3_num1 = im_arr_C2_E3_test[5].reshape(1, 70, 70, 1)

C0_test_l_outputs = activation_model.predict([C0_test_E0_num1, C0_test_E1_num1, C0_test_E2_num1, C0_test_E3_num1])
C1_test_l_outputs = activation_model.predict([C1_test_E0_num1, C1_test_E1_num1, C1_test_E2_num1, C1_test_E3_num1])
C2_test_l_outputs = activation_model.predict([C2_test_E0_num1, C2_test_E1_num1, C2_test_E2_num1, C2_test_E3_num1])

print ('check types from layers: ', type(C0_test_l_outputs) )

print ('what is in the list: check len and first element and shape', len(C0_test_l_outputs), C0_test_l_outputs[0], 
       '\n', C0_test_l_outputs[1].shape, '\n', C0_test_l_outputs[2].shape, '\n', 
       C0_test_l_outputs[5].shape)

###########################################
#
### from keras plots we check the layer names and correspondng index that we want to plot
# this would change if the model changes 
# 1st conv for each enrgy bin [5, 6, 7, 8]
# 2nd conv for each energy bin [9, 10, 11, 12]
# 3rd conv we choose after concatenation
# [14], [16]
#########################################

conv_number1, conv_number2, conv_number3, conv_number4 = 6, 24, 36, 48 # we have 64 layers here 
chosen_layers = [16]

name_layers ='conv4'


def subplots_conv_out(im_outs, ext_cls, rows=len(chosen_layers), cols=4):
    fig = plt.figure(figsize=(16, 5), )
    f1 = im_outs[chosen_layers[0]]
    fig.add_subplot(141)
    im1_C0 = plt.imshow(f1[0, :, :, conv_number1], cmap='inferno')
    plt.title('Filter: %d'%(conv_number1))
    plt.colorbar(shrink=0.55)
    fig.add_subplot(142)
    im2_C0 = plt.imshow(f1[0, :, :, conv_number2], cmap='inferno')
    plt.title('Filter: %d'%(conv_number2))
    plt.colorbar(shrink=0.55)
    fig.add_subplot(143)
    im3_C0 = plt.imshow(f1[0, :, :, conv_number3], cmap='inferno')
    plt.title('Filter: %d'%(conv_number3))
    plt.colorbar(shrink=0.55)
    fig.add_subplot(144)
    im4_C0 = plt.imshow(f1[0, :, :, conv_number4], cmap='inferno')
    plt.title('Filter: %d'%(conv_number4))
    plt.colorbar(shrink=0.55)
    plt.tight_layout() # rect=[0, 0.03, 1, 0.95])# rect [l, b, r, top]        
    fig.suptitle('%s'%(name_layers), fontsize=11, y=0.8) 
    fig.savefig(homepath + 'Images/plots/check_conv_layers_C%d_999_final.png'%(ext_cls), dpi=200, bbox_inches='tight')

subplots_conv_out(C0_test_l_outputs, 0)
subplots_conv_out(C1_test_l_outputs, 1)
subplots_conv_out(C2_test_l_outputs, 2)
