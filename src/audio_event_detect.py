"""
Author:   KaiWu Wang
Created:  11/11/2017
"""
from __future__ import print_function 
import sys, os, time, argparse, tqdm
import cPickle, glob, gzip, csv, h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pdb

import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Multiply

from prepare_data import create_folder, load_hdf5_data, do_scale, second_to_frame, read_yaml
from data_generator import MyGenerator as Generator
import config as cfg


def my_plot(pd, save_path, gt):
    folder,_ = os.path.split(save_path)
    if not os.path.exists(folder):
        create_folder(folder)
    for i in range(4):
        plt.subplot(221+i)
        plt.plot(range(240), gt[:,i], 'r')
        plt.bar(left=range(240), height=pd[:,i], width=1, color='b')
        plt.xlim(0, 251)
        plt.ylim(0, 1.1)
    plt.savefig(save_path)
    #plt.show()
    plt.close()


def block(input):
    cnn = Conv2D(128, (1, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])


def train(args):
    num_classes = cfg.num_classes
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    # (tr_x, tr_y, tr_na_list) = load_hdf5(args.te_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("")

    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    
    # Build model
    (_, n_time, n_freq) = tr_x.shape
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')
    a1 = Reshape((n_time, n_freq, 1))(input_logmel)
    
    cnn1 = block(a1)
    cnn1 = block(cnn1)
    cnn1 = MaxPooling2D(pool_size=(1, 2))(cnn1)
    
    cnn2 = block(cnn1)
    cnn2 = block(cnn2)
    cnn2 = MaxPooling2D(pool_size=(1, 2))(cnn2)
    
    cnn3 = block(cnn2)
    cnn3 = block(cnn3)
    cnn3 = MaxPooling2D(pool_size=(1, 2))(cnn3)
    
    cnn4 = block(cnn3)
    cnn4 = block(cnn4)
    cnn4 = MaxPooling2D(pool_size=(1, 2))(cnn4)
    
    cnnout = Conv2D(256, (1, 3), padding="same", activation="relu", use_bias=True)(cnn4)
    cnnout = MaxPooling2D(pool_size=(1, 4))(cnnout)
    
    cnnout = Reshape((240, 256))(cnnout)   # Time step is downsampled to 30. 
    
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(cnnout)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(cnnout)
    out = Multiply()([rnnout, rnnout_gate])
    
    #out = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='output_layer')(out)
    out = TimeDistributed(Dense(num_classes, activation='softmax'), name='output_layer')(out)
    #det =TimeDistributed(Dense(num_classes, activation='softmax'))(out)
    #out=Multiply()([out,det])
    #out=Lambda(outfunc, output_shape=(num_classes,))([out, det])
    
    model = Model(input_logmel, out)
    model.summary()
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, "aed_batchsize50_lr0.001_LogMel64_{epoch:04d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = Generator(batch_size=50, type='train')

    # Train
    model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=300,    # 100 iters is called an 'epoch'
                        epochs=100,      # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_x, te_y))


# Recognize and write probabilites. 
def recognize(args):
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list
    
    x = do_scale(x, args.scaler_path, verbose=1)
    fusion_at_list = []
    fusion_sed_list = []
    for epoch in range(10, 11, 1):
        t1 = time.time()
        [model_path] = glob.glob(os.path.join(args.model_dir, 
            "*%04d-0.*.hdf5" % epoch))
        model = load_model(model_path)
        # Audio tagging
        pred = model.predict(x)
        fusion_at_list.append(pred)
        
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    fusion_at = np.mean(np.array(fusion_at_list), axis=0)
    print("AT shape: %s" % (fusion_at.shape,))
    for audio_ind in range(fusion_at.shape[0]):
        save_path=os.path.join("result","pictures",na_list[audio_ind].replace("wav","jpg"))
        my_plot(fusion_at[audio_ind,...], save_path, y[audio_ind,...])
        
    print("Prediction finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--te_hdf5_path', type=str)
    parser_recognize.add_argument('--scaler_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--yaml_dir', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'recognize':
        recognize(args)
    else:
        raise Exception("Incorrect argument!")
