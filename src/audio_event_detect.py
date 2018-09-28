"""
Author:   KaiWu Wang
Created:  11/11/2017
"""
from __future__ import print_function 
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cPickle, glob, gzip, csv, h5py
import sys, os, time, argparse, tqdm, shutil
import pdb

import keras
import tensorflow as tf
from keras import backend as K
from keras import optimizers, regularizers, losses

from keras.layers import Input, TimeDistributed, Bidirectional
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Multiply, Add
from keras.layers.convolutional import ZeroPadding2D, Conv2D, Conv1D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling1D, MaxPooling1D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.models import Sequential, Model, load_model

import config as cfg
from data_generator import MyGenerator as Generator
from prepare_data import create_folder, load_hdf5, do_scale, second_to_frame, read_yaml


def pro_boundary(pd, threshold, mg, ig):
    pd_pro = np.zeros(pd.shape, dtype=int)
    pd_pre = np.zeros(pd.shape, dtype=int)
    inds = np.where(pd > threshold)[0]
    segments = []
    if not inds.shape[0] == 0:
        #pdb.set_trace()
        pd_pro[inds] = 1
        pd_pre[1:] = pd_pro[:-1]
        end_ = 0
        if pd_pro[-1] == 1:
           end_ = 1
        pd_pro -= pd_pre
        starts = np.where(pd_pro == 1)[0]
        ends = np.where(pd_pro == -1)[0] -1
        if end_ == 1:
            ends = np.concatenate( (ends, [pd.shape[0]-1]) )
        if not starts.shape == ends.shape:
            return segments
        for i in range(0,starts.shape[0]):
            if len(segments)>0 and starts[i] - segments[-1][-1] < mg:
                segments[-1][-1]=ends[i]
                continue
            if ends[i]-starts[i] > ig:
                segments.append([starts[i], ends[i]])
    return segments


def my_plot(pd, gt, picture_path, threshold=None):
    classes = cfg.classes
    ig = cfg.ig
    mg = cfg.mg

    estimate_path = picture_path.replace("picture", "estimate_txt")
    estimate_path = estimate_path.replace("jpg", "txt")

    folder,_ = os.path.split(picture_path)
    if not os.path.exists(folder):
        create_folder(folder)

    folder,_ = os.path.split(estimate_path)
    if not os.path.exists(folder):
        create_folder(folder)

    result=open(estimate_path,'at')
    n_cls = len(classes)
    if threshold==None:
        pd_ = pd.argmax(axis=-1)
    for i in range(n_cls):
        #'''
        plt.subplot(221+i)
        plt.plot(range(240), gt[:,i], 'r')
        plt.bar(left=range(240), height=pd[:,i], width=1, color='b')
        plt.xlim(0, 251)
        plt.ylim(0, 1.1)
        #'''
        if not i==0:
            if threshold==None:
                class_ind = np.where(pd_==i)[0]
                pd_class=np.zeros(pd_.shape)
                pd_class[class_ind]=1
                segments = pro_boundary(pd_class, 0, mg[i], ig[i])
            else:
                segments = pro_boundary(pd[:,i], threshold[i], mg[i], ig[i])
            for j in range(len(segments)):
                #'''
                plt.plot([segments[j][0]]*240, np.arange(240)/240.0*1.1, 'g')
                plt.plot([segments[j][1]]*240, np.arange(240)/240.0*1.1, 'g')
                #'''
                result.write(str(segments[j][0]*cfg.step_time)+'\t' + 
 		    		str(segments[j][1]*cfg.step_time)+'\t' + 
								classes[i]+'\n')
    #'''
    plt.savefig(picture_path)
    #plt.show()
    plt.close()
    #'''
    result.close()


def Conv_BN(input, k, kernel, pad="same", act="relu", bias=False):
    out = Conv2D(k, kernel, padding=pad, activation="linear", use_bias=bias, 
				kernel_regularizer=regularizers.l2(0.0))(input)
    out = BatchNormalization(axis=-1)(out)
    out = Activation(act)(out)
    return out


def block_a(input, k):
    cnn_c = Conv_BN(input, k, (3, 1), act="sigmoid")
    cnn_r = Conv_BN(input, k, (1, 3), act="relu")
    out = Multiply()([cnn_r, cnn_c])
    return out


def block_b(input, k):
    cnn_c = Conv_BN(input, k, (3, 1), act="sigmoid")
    cnn_r = Conv_BN(input, k, (1, 3), act="relu")
    cnn_r = Conv_BN(cnn_r, k, (1, 3), act="relu")
    out = Multiply()([cnn_r, cnn_c])
    return out


def block_c(input, k):
    '''
    out = Conv_BN(input, k, (3, 3), act="relu")
    out = block_a(input, k)
    '''
    out = block_b(input, k)
    out = Add()([input, out])
    out = Conv_BN(out, 64, (1, 1), act="relu")
    return out


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_fixed(y_true, y_pred):
        #pdb.set_trace()
        '''
        return -K.mean(K.mean(K.sum(y_true * K.log(y_pred + K.epsilon()), axis=-1)))
        '''
        if cfg.num_classes>1:
            pt = K.sum(y_true*y_pred, axis=-1)
            alpha_ = K.sum(y_true*alpha, axis=-1)
        else:
            pt = K.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_ = K.where(K.equal(y_true, 1), alpha, 1 - alpha)
        return -K.mean(K.mean(alpha_ * K.pow(1. + K.epsilon() - pt, gamma) * K.log(pt+K.epsilon())))
    return focal_loss_fixed


def myacc(threshold=0.5):
    def Acc(y_true, y_pred):
        #pdb.set_trace()
        return K.mean(K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())))
        #return K.mean(K.mean(K.cast(K.sum(y_true*y_pred, axis=-1)>threshold, K.floatx())))
    return Acc


def train(args):
    if os.path.exists(args.out_model_dir):
        shutil.rmtree(args.out_model_dir)
    create_folder(args.out_model_dir)
    num_classes = cfg.num_classes
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5(args.tr_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5(args.te_hdf5_path, verbose=1)
    print("")

    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    # Build model
    (_, n_time, n_freq) = tr_x.shape

    #pdb.set_trace()

    input = Input(shape=(n_time, n_freq), name='input_layer')
    input_ = Reshape((n_time, n_freq, 1))(input)

    '''
    block1 = Conv_BN(input_, 8, (3, 3), act="relu")
    block1 = Conv_BN(block1, 32, (3, 3), act="relu")
    block1 = Conv_BN(block1, 64, (3, 3), act="relu")

    block1 = block_a(input_, 8)
    block1 = block_a(block1, 32)
    block1 = block_a(block1, 64)
    '''
    block1 = block_b(input_, 8)
    block1 = block_b(block1, 32)
    block1 = block_b(block1, 64)
    block1 = MaxPooling2D(pool_size=(1, 2))(block1)

    block2 = block_c(block1, 64)
    block2 = MaxPooling2D(pool_size=(1, 2))(block2)

    block3 = block_c(block2, 64)
    block3 = MaxPooling2D(pool_size=(1, 2))(block3)
    
    block4 = block_c(block3, 64)
    block4 = MaxPooling2D(pool_size=(1, 2))(block4)    

    cnnout = Conv_BN(block4, 128, (1, 1), act="relu", bias=True)
    cnnout = MaxPooling2D(pool_size=(1, 2))(cnnout)
    cnnout = Reshape((240, 256))(cnnout) 
    
    rnn = Bidirectional(GRU(128, activation='relu', return_sequences=True, 
						kernel_regularizer=regularizers.l2(0.01), 
						recurrent_regularizer=regularizers.l2(0.01)))(cnnout)

    out = TimeDistributed(Dense(num_classes, activation='softmax', 
				kernel_regularizer=regularizers.l2(0.0), ),name='output_layer')(rnn)

    model = Model(input, out)
    model.summary()

    # Compile model
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.009)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0)
    model.compile(loss=focal_loss(alpha=[1,1,1,1], gamma=1),
                  optimizer="adam",
                  metrics=[myacc(threshold=0.5)]
		  )
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, "aed-batchsize_50-lr_0.01-{epoch:04d}-{val_Acc:.4f}.hdf5")
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_Acc',
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Train
    '''
    history=model.fit(  x=tr_x, 
			y=tr_y, 
			batch_size=50, 
			epochs=200, 
			verbose=1,
			shuffle=True,
			class_weight="auto", 
			callbacks=[save_model], 
			validation_data=(te_x,te_y)
		      ) 

    '''
    # Data generator
    gen = Generator(batch_size=50, type='train')
    history=model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=300,    # 100 iters is called an 'epoch'
                        epochs=100,      # Maximum 'epoch' to train
                        verbose=1, 
			class_weight="auto", 
                        callbacks=[save_model], 
                        validation_data=(te_x, te_y))

    with open('src/log.py','w') as f:
        f.write("history=")
        f.write(str(history.history))


# detect 
def detect(args):
    import log
    val_loss = log.history["val_loss"]
    val_acc = log.history["val_Acc"]
    choose=[]
    for i in range(1):
        '''
        min_loss = np.argmin(val_loss)
        choose.append(min_loss)
        val_loss[min_loss] = np.inf
        '''
        max_acc = np.argmax(val_acc)
        choose.append(max_acc)
        val_acc[max_acc] = 0

    (te_x, te_y, te_na_list) = load_hdf5(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list

    x = do_scale(x, args.scaler_path, verbose=1)
    fusion_at_list = []
    fusion_sed_list = []
    #choose=[48]
    for epoch in choose:
        t1 = time.time()
        [model_path] = glob.glob(os.path.join(args.model_dir, "*-%04d-*hdf5" % epoch))
        model = load_model(model_path,custom_objects={'focal_loss_fixed': focal_loss(), 'Acc': myacc()})
        print("load the model: %s" % model_path)
        # Audio tagging
        pred = model.predict(x)
        fusion_at_list.append(pred)
        
        print("Prediction time: %s" % (time.time() - t1,))
    # Write out AT probabilities
    fusion_at = np.mean(np.array(fusion_at_list), axis=0)
    print("AT shape: %s" % (fusion_at.shape,))
    if os.path.exists("result"):
        shutil.rmtree("result")

    for audio_ind in range(fusion_at.shape[0]):
        #if na_list[audio_ind]=="mixture_babycry_0.0_0016_65021d74d0fb56db84b63896e2ff5ec9.wav":
        #    picture_path=os.path.join("result","picture",na_list[audio_ind].replace("wav","jpg"))
        #    pdb.set_trace()
        #    my_plot(fusion_at[audio_ind,...], y[audio_ind,...], picture_path)
        picture_path=os.path.join("result","picture",na_list[audio_ind].replace("wav","jpg"))
        #my_plot(fusion_at[audio_ind,...], y[audio_ind,...], picture_path)
        my_plot(fusion_at[audio_ind,...], y[audio_ind,...], picture_path, threshold=cfg.threshold)
        
    print("Prediction finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="audio event detect")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)
    
    parser_detect = subparsers.add_parser('detect')
    parser_detect.add_argument('--te_hdf5_path', type=str)
    parser_detect.add_argument('--scaler_path', type=str)
    parser_detect.add_argument('--model_dir', type=str)
    parser_detect.add_argument('--yaml_dir', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'detect':
        detect(args)
    else:
        raise Exception("Incorrect argument!")
