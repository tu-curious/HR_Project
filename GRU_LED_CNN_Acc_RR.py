# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:34:06 2018

@author: agarwal.270
"""
import pandas as pd
import timeit
import peakutils as pk
import time
import os
import glob
from pylab import *
from Tompkin_modified_GRU import detect_rpeak as dr_modified
from Tomikn import detect_rpeak
import scipy.signal as sig
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
#set_random_seed(1)
import keras as kr
from keras.models import Model,Sequential # Neural-Network model
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import MaxPooling2D, Input, concatenate, Lambda, Reshape
from keras.layers import Conv1D, TimeDistributed, LSTM, GRU , Bidirectional
from keras.layers.normalization import BatchNormalization
import keras.regularizers as regularizers
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
#from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import keras.backend as K


# In[]
def visualeyes(arr):
    figure()
    ax=subplot(211)
    plot(arr[:,-1],'b')
    hold(True)
    plot(arr[:,3],'r')
    hold(True)
    plot(arr[:,4],'y')
    hold(True)
    plot(arr[:,5],'g')
    legend(['ECG (filtered)','LED_R','LED_I','LED_G'])
    title('ECG & LED Data ('+str(len(arr))+' ECG Samples)')
    grid(True)
    subplot(212,sharex=ax)
    plot(arr[:,0],'r')
    hold(True)
    plot(arr[:,1],'g')
    hold(True)
    plot(arr[:,2],'b')
    legend(['ECG','AccX','AccY','AccZ'])
    title('Acc Data')
    grid(True)
    return None

def norma(df):
    std=df.std();tol=1e-10
    for i in range(len(std)):
        if abs(std[i])<=tol:
            print(std.index[i]+' is saturated')
            std[i]=inf
    dfnor=(df-df.mean())/std
    return dfnor

def HR_predict(arr,w,stryd,Fs=25,filt=False):
    arr0=np.array([list(arr[stryd*i:(stryd*i)+w]) for i in np.arange(np.round((len(arr)-w+1)/(stryd))).astype(int)])
    HR_ruf=arr0.sum(axis=1)/(w*(1/Fs)/60) # HR in BPM
    if filt:
        HR_smooth=sig.savgol_filter(HR_ruf,((Fs*10+1)//stryd),2)
        HR=HR_smooth
    else:
        HR=HR_ruf
    return HR

def seg_tensor(input_layer,params):
    C11=Conv1D(filters=params[0],kernel_size=params[1],strides=params[2],padding='same')(input_layer)
    B01=BatchNormalization()(C11)
    A01=Activation('relu')(B01)
    D01=Dropout(0.2)(A01)
    return D01


def exp_loss(y_true,y_pred):
    return K.mean(K.exp(5.+K.square(y_pred - y_true)), axis=-1)
# In[]
#figure();plot(np.arange(0,1,0.01),np.exp(np.arange(0,1,0.01)**2+5),'b',np.arange(0,1,0.01),(np.arange(0,1,0.01)**2),'r')
 # Load time-series Data
#import data
path_list=[]
path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Jan18_Week5\\Data_Prof_EE\\raw\\extracted_data_LSTM_RR\\']
path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Jan18_Week5\\Data_Nithin\\raw\\extracted_data_LSTM_RR\\']
path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Jan18_Week5\\Data_Diyan\\raw_motion\\extracted_data_LSTM_RR\\']
#path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Feb_Week1\\data_tushar\\training_data1\\']
#path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Feb_Week1\\data_tushar\\training_data2\\']
#path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Feb_Week1\\data_tushar\\training_data3\\']
#path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Feb_Week1\\data_tushar\\training_data4\\']
#path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Feb_Week1\\data_tushar\\training_data5\\']
#path_list=path_list+['C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Feb_Week1\\data_tushar\\training_data6\\']
# Load one long sequence


list_arr_X_RA=[];list_arr_Y_RA=[];list_arr_X_LA=[];list_arr_Y_LA=[]
for path in path_list:
    n=1
    while os.path.exists(path+'ECG'+str(n)+'.csv'):
        arr_ECG=pd.read_csv(path+'ECG'+str(n)+'.csv').values[:,1].astype(float)
        #arr_RA=pd.read_csv(path+'RA'+str(n)+'.csv').values[:,[0,1,2,6,7,8,9]].astype(float)
        df_RA=pd.read_csv(path+'RA'+str(n)+'.csv').set_index('time')
        df_LA=pd.read_csv(path+'LA'+str(n)+'.csv').set_index('time')
        df_RA.index=pd.to_datetime(df_RA.index)
        df_LA.index=pd.to_datetime(df_LA.index)
                
        # resample RA at LA times
        df_RA_tmp1=df_LA.merge(df_RA,how='outer',left_index=True,right_index=True,sort=True,suffixes=('_LA','_RA'))
        df_RA_tmp1.interpolate(method='time',axis=0,inplace=True)
        df_RA_tmp1.dropna(inplace=True)
        df_RA_tmp1=df_RA_tmp1.loc[df_LA.index]
        df_RA_tmp1=df_RA_tmp1[[col for col in df_RA_tmp1.columns if '_RA' in col]]
        
        arr_RA_old=df_RA.values[:,[0,1,2,6,7,8,9]].astype(float)
        arr_LA=df_LA.values[:,[0,1,2,6,7,8,9]].astype(float)
        arr_RA=df_RA_tmp1.values[:,[0,1,2,6,7,8,9]].astype(float)
# =============================================================================
#         figure()
#         plot(arr_RA_old[:,-2]);hold(True);plot(arr_RA[:,-2],'--');grid(True)
#         legend(['Original LED','Resampled LED'])
# =============================================================================
        
        
        # feed to CNN model to get both probabilities

        #prob_RA,prob_LA=CNN_predict(final_model,arr_RA_old[:,:-1],arr_LA[:,:-1])
        
        # feed these 2 probabilities into the model
        w=11;
        #arr_RA=np.vstack([prob_RA,arr_RA[:-(w-1),-1]]).T
        #arr_LA=np.vstack([prob_LA,prob_RA,arr_LA[:-(w-1),-1]]).T
        arr_RA_new=np.concatenate([arr_RA_old[:-(w-1),:-1],arr_RA_old[:-(w-1),-1].reshape(-1,1)],axis=1)
        arr_LA_new=np.concatenate([arr_LA[:-(w-1),:-1],arr_LA[:-(w-1),-1].reshape(-1,1)],axis=1)
        #drop any Nans
        arr_RA_new=arr_RA_new[~np.isnan(arr_RA_new).any(axis=1)]
        arr_LA_new=arr_LA_new[~np.isnan(arr_LA_new).any(axis=1)]
        
        
        batch_size=4; st=200 #stride
        Tx=batch_size*st # No. of time-steps (important heuristic)
        idxes_RA=np.arange(0,arr_RA_new.shape[0]-Tx,st).astype(int)  # change Tx or st for overlap
        for idx in idxes_RA:
            if idx==0:
                arr_concat=arr_RA_new[idx:idx+Tx,:].reshape(1,Tx,-1)
            else:
                arr_concat=np.concatenate([arr_concat,arr_RA_new[idx:idx+Tx,:].reshape(1,Tx,-1)])
        arr_X_RA,arr_Y_RA=arr_concat[:,:,:-1],arr_concat[:,:,-1].reshape(-1,Tx,1)
        del arr_concat
        list_arr_X_RA=list_arr_X_RA+[arr_X_RA];list_arr_Y_RA=list_arr_Y_RA+[arr_Y_RA]

        idxes_LA=np.arange(0,arr_LA_new.shape[0]-Tx,st).astype(int)  # change Tx or st for overlap
        for idx in idxes_LA:
            if idx==0:
                arr_concat=arr_LA_new[idx:idx+Tx,:].reshape(1,Tx,-1)
            else:
                arr_concat=np.concatenate([arr_concat,arr_LA_new[idx:idx+Tx,:].reshape(1,Tx,-1)])
        arr_X_LA,arr_Y_LA=arr_concat[:,:,:-1],arr_concat[:,:,-1].reshape(-1,Tx,1)
        del arr_concat
        list_arr_X_LA=list_arr_X_LA+[arr_X_LA];list_arr_Y_LA=list_arr_Y_LA+[arr_Y_LA]
        n=n+1
        
#sess_all.close()   
# merge both RA and LA
list_arr_X=list_arr_X_RA+list_arr_X_LA
list_arr_Y=list_arr_Y_RA+list_arr_Y_LA

# concatenate all arrays for the stateless model
arr_X=np.concatenate(list_arr_X,axis=0);arr_Y=np.concatenate(list_arr_Y,axis=0)
p=np.random.permutation(arr_X.shape[0]);arr_X=arr_X[p];arr_Y=arr_Y[p] #shuffle
val_split=0.1;lenth=int(val_split*arr_X.shape[0])
val_X,val_Y=arr_X[:lenth],arr_Y[:lenth];train_X,train_Y=arr_X[lenth:],arr_Y[lenth:]

#del list_arr_X[6],list_arr_X[1],list_arr_Y[6],list_arr_Y[1]

# In[]
fil=64;fil_len=15;stryd=1;Tx_C=int((Tx-fil_len+1)/stryd)
params=[fil,fil_len,stryd]
input_dim=(arr_X_RA.shape[1:-1])

input_layer0=Input(shape=(input_dim)+(1,))
input_layer1=Input(shape=(input_dim)+(1,))
input_layer2=Input(shape=(input_dim)+(1,))
input_layer3=Input(shape=(input_dim)+(3,))
#input_layer=tf.expand_dims(input_layer[:,:,0],2)
#input1=Reshape((Tx,1))(input_layer[:,:,0])

D00 = seg_tensor(input_layer0,params)
D01 = seg_tensor(input_layer1,params)
D02 = seg_tensor(input_layer2,params)

#input3 = Lambda(lambda x: x[:,:,3:])(input_layer)
C03=Conv1D(filters=fil,kernel_size=fil_len,strides=stryd,padding='same')(input_layer3)
B03=BatchNormalization()(C03)
A03=Activation('relu')(B03)
D03=Dropout(0.2)(A03)

merge1=concatenate([D00,D01,D02])
#print(kr.backend.int_shape(merge1))

G1=Bidirectional(GRU(128, activation='tanh',\
       return_sequences=True, stateful=False, unroll=False))(merge1)
D1=Dropout(0.2)(G1)
B1=BatchNormalization()(D1)

G2=Bidirectional(GRU(64, activation='tanh',return_sequences=True,\
       stateful=False, unroll=False))(B1)
D2=Dropout(0.2)(G2)
B2=BatchNormalization()(D2)

G3=Bidirectional(GRU(32, activation='tanh',return_sequences=True,\
       stateful=False, unroll=False))(B2)
D3=Dropout(0.2)(G3)
FC1=TimeDistributed(Dense(1))(D3)
#B3=BatchNormalization()(D3)

C04=Conv1D(filters=fil,kernel_size=fil_len,strides=stryd,padding='same')(FC1)
B04=BatchNormalization()(C04)
A04=Activation('relu')(B04)

merge2=concatenate([A04,D03])

C05=Conv1D(filters=64,kernel_size=fil_len,strides=stryd,padding='same')(merge2)
B05=BatchNormalization()(C05)
A05=Activation('relu')(B05)
C06=Conv1D(filters=32,kernel_size=fil_len,strides=stryd,padding='same')(A05)
B06=BatchNormalization()(C06)
A06=Activation('relu')(B06)
D06=Dropout(0.2)(A06)

C07=Conv1D(filters=1,kernel_size=fil_len,strides=stryd,padding='same')(D06)



#Flat1=Flatten()(A06)







# =============================================================================
# print(kr.backend.int_shape(FC1))
# #Deconv
# FC1=Lambda(lambda x: tf.expand_dims(x,1))(FC1)
# print(kr.backend.int_shape(FC1))
# DC1=kr.layers.Conv2DTranspose(filters=1,kernel_size=(1,fil_len+stryd-1),strides=(1,stryd))(FC1)
# print(kr.backend.int_shape(DC1))
# FC2=Lambda(lambda x: tf.squeeze(x,1))(DC1)
# =============================================================================

#FC2=TimeDistributed(Dense(1))(FC1)
GRU_model=Model([input_layer0,input_layer1,input_layer2,input_layer3],C07)
#opt = kr.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
# =============================================================================
# LS1_model=Sequential()
# LS1_model.add(LSTM(100, batch_input_shape=(1,Tx,6),activation='tanh', recurrent_activation='hard_sigmoid',\
#          use_bias=True, kernel_initializer='glorot_uniform', \
#          recurrent_initializer='orthogonal', bias_initializer='zeros', \
#          unit_forget_bias=True,dropout=0.0, recurrent_dropout=0.0, \
#          return_sequences=True, return_state=False, go_backwards=False, \
#          stateful=True, unroll=False))
# LS1_model.add(TimeDistributed(Dense(1)))
# =============================================================================

#GRU_model.load_weights('model_OSUdata_3GRU_Tr2.h5')
GRU_model.compile(loss='mse', optimizer='adam',metrics=['mse'])
print(GRU_model.summary())
plot_model(GRU_model, to_file='GRU_LED_Acc_model.png',show_shapes=True)


loss_list=[];elist=[];val_loss_list=[];
# In[Train]
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# Ask keras to use particular device for training
config=tf.ConfigProto(device_count={'GPU':1,'CPU':1},log_device_placement=True)
sess_all=tf.Session(config=config)
#sess_all.run(tf.global_variables_initializer())
kr.backend.set_session(sess_all)

# checkpoint
filepath="model_OSUdata_GRU_LEDAcc_RR_Tr2.h5"
Mcheck = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
#Echeck=kr.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0)

callbacks_list = [Mcheck]

#make figure for plotting loss
ini_epoch=len(elist)
plt.ion();fig = plt.figure();ax = fig.add_subplot(111);
plt.title('Training Loss (mse)');plt.xlabel('Epoch');plt.ylabel('mse');plt.grid(True)
for i in range(220):    # No. of Epochs
    histry=GRU_model.fit([train_X[:,:,3:4],train_X[:,:,4:5],train_X[:,:,5:6],train_X[:,:,0:3]],\
                         train_Y,epochs=1,batch_size=270,\
                         validation_data=([val_X[:,:,3:4],val_X[:,:,4:5],val_X[:,:,5:6],val_X[:,:,0:3]],val_Y)\
                         ,verbose=2,callbacks=callbacks_list)
    loss_list=loss_list+[histry.history['loss'][0]];elist=elist+[i+ini_epoch]
    val_loss_list=val_loss_list+[histry.history['val_loss'][0]]
    ax.plot(elist,loss_list,'b',elist,val_loss_list,'r--')
    ax.legend(['Train_Loss','Val_Loss']);fig.canvas.draw();plt.pause(0.05)
      
#sess_all.close()
# In[]
        
GRU_model=load_model("model_OSUdata_3GRU_RR_Tr1.h5");batch_size=256

print(GRU_model.summary())
plot_model(GRU_model, to_file='GRU_model_test.png',show_shapes=True)

pathtest='C:\\Users\\agarwal.270\\Box Sync\\Research@ECE\\PPG_ECG_Project\\Shared Folder\\Jan18_Week5\\Data_Nithin\\raw\\extracted_data_LSTM_RR\\'
n=1


df_test_RA=pd.read_csv(pathtest+'test_RA'+str(n)+'.csv').set_index('time')
df_test_LA=pd.read_csv(pathtest+'test_LA'+str(n)+'.csv').set_index('time')
df_test_RA.index=pd.to_datetime(df_test_RA.index)
df_test_LA.index=pd.to_datetime(df_test_LA.index)


arr_RA=df_test_RA.values[:,[0,1,2,6,7,8,9]].astype(float)
arr_LA=df_test_LA.values[:,[0,1,2,6,7,8,9]].astype(float)

# =============================================================================
#         figure()
#         plot(arr_RA_old[:,-2]);hold(True);plot(arr_RA[:,-2],'--');grid(True)
#         legend(['Original LED','Resampled LED'])
# =============================================================================

# =============================================================================
# # Ju's data comparison
# arr_RA=pd.read_csv(pathtest+'data_compare_R.csv',header=None).values
# arr_RA=np.concatenate([arr_RA,np.zeros((arr_RA.shape[0],1))],axis=1)
# arr_LA=pd.read_csv(pathtest+'data_compare_L.csv',header=None).values
# arr_LA=np.concatenate([arr_LA,np.zeros((arr_LA.shape[0],1))],axis=1)
# =============================================================================

# feed to CNN model to get both probabilities

#prob_RA,prob_LA=CNN_predict(final_model,arr_RA[:,:-1],arr_LA[:,:-1])

# feed these 2 probabilities into the model
w=11;
#arr_RA=np.vstack([prob_RA,arr_RA[:-(w-1),-1]]).T
#arr_LA=np.vstack([prob_LA,prob_RA,arr_LA[:-(w-1),-1]]).T

#arr_RA_new=np.concatenate([arr_RA[:-(w-1),:-1],prob_RA,arr_RA[:-(w-1),-1].reshape(-1,1)],axis=1)
#arr_LA_new=np.concatenate([arr_LA[:-(w-1),:-1],prob_LA,arr_LA[:-(w-1),-1].reshape(-1,1)],axis=1)
#drop any Nans
#arr_RA=arr_RA[~np.isnan(arr_RA).any(axis=1)]
arr_RA_new=arr_RA[~np.isnan(arr_RA).any(axis=1)]
arr_LA_new=arr_LA[~np.isnan(arr_LA).any(axis=1)]


Fs=25;batch_size=4; st=200 #stride
Tx=batch_size*st # No. of time-steps (important heuristic)
st_RR=Tx-(20*Fs) #20 s of overlap as in original
idxes_RA=np.arange(0,arr_RA_new.shape[0]-Tx,Tx).astype(int)  # change Tx or st for overlap
for idx in idxes_RA:
    if idx==0:
        arr_concat=arr_RA_new[idx:idx+Tx,:].reshape(1,Tx,-1)
    else:
        arr_concat=np.concatenate([arr_concat,arr_RA_new[idx:idx+Tx,:].reshape(1,Tx,-1)])
arr_Xt_RA,arr_Yt_RA=arr_concat[:,:,:-1],arr_concat[:,:,-1].reshape(-1,Tx,1)
del arr_concat

idxes_LA=np.arange(0,arr_LA_new.shape[0]-Tx,Tx).astype(int)  # change Tx or st for overlap
for idx in idxes_LA:
    if idx==0:
        arr_concat=arr_LA_new[idx:idx+Tx,:].reshape(1,Tx,-1)
    else:
        arr_concat=np.concatenate([arr_concat,arr_LA_new[idx:idx+Tx,:].reshape(1,Tx,-1)])
arr_Xt_LA,arr_Yt_LA=arr_concat[:,:,:-1],arr_concat[:,:,-1].reshape(-1,Tx,1)
del arr_concat
    
y_hat_RA=GRU_model.predict([arr_Xt_RA[:,:,3:4],arr_Xt_RA[:,:,4:5],arr_Xt_RA[:,:,5:6],arr_Xt_RA[:,:,0:3]])
y_hat_LA=GRU_model.predict([arr_Xt_LA[:,:,3:4],arr_Xt_LA[:,:,4:5],arr_Xt_LA[:,:,5:6],arr_Xt_LA[:,:,0:3]])

#Acc_RMS=np.sqrt(np.sum(arr_Xt_RA[:,:,3:]**2,axis=-1))
#arr_Yt_RA=arr_Yt_RA[:y_hat_RA.shape[0],:,:]

# Unwrap the sequence
Y_pred_RA=y_hat_RA.reshape(-1)
Y_true_RA=arr_Yt_RA.reshape(-1)
Y_pred_LA=y_hat_LA.reshape(-1)
Y_true_LA=arr_Yt_LA.reshape(-1)

# =============================================================================
# #masking waste elements
# mask_RA=np.ones(len(Y_pred_RA),dtype=bool)
# for llidx in np.arange(Tx,len(Y_pred_RA),Tx):
#     mask_RA[llidx:llidx+(20*Fs)]=False
# 
# Y_pred_RA=Y_pred_RA[mask_RA]
# Y_true_RA=Y_true_RA[mask_RA]
# 
# #masking waste elements
# mask_LA=np.ones(len(Y_pred_LA),dtype=bool)
# for llidx in np.arange(Tx,len(Y_pred_LA),Tx):
#     mask_LA[llidx:llidx+(20*Fs)]=False
# 
# Y_pred_LA=Y_pred_LA[mask_LA]
# Y_true_LA=Y_true_LA[mask_LA]
# =============================================================================

#weighting based on RMS
Acc_rms_LA=(np.sum(arr_LA[:len(Y_pred_LA),0:3]**2,axis=1))**0.5
Acc_rms_RA=(np.sum(arr_RA[:len(Y_pred_RA),0:3]**2,axis=1))**0.5
figure();plot(Acc_rms_LA);hold(True);plot(Acc_rms_RA);grid(True);legend(['LA','RA'])

z_LA=np.absolute(arr_LA[:len(Y_pred_LA),5])
z_RA=np.absolute(arr_RA[:len(Y_pred_RA),5])
figure();plot(z_LA);hold(True);plot(z_RA);grid(True);legend(['LA','RA'])


figure()
ax=subplot(211) 
plot(Y_true_RA);hold(True);plot(Y_pred_RA,'r--');grid(True)
title('Right Arm');legend(['True','Predicted'])
subplot(212,sharex=ax,sharey=ax)
plot(Y_true_LA);hold(True);plot(Y_pred_LA,'r--');grid(True)
title('Left Arm');legend(['True','Predicted'])

# Run peak detection
Fs=25;max_HR=180 #BPM
#ind = pk.indexes(Y_true, thres=0.2, min_dist=round(Fs/(max_HR/60)))
ind = dr_modified(list(Y_true_RA),Fs,0.2)
ser_Y_true=pd.Series(Y_true_RA)
#ser_ECG.index=ser_ECG.index*0.25
figure()
plot(ser_Y_true);hold(True);plot(ser_Y_true.iloc[ind],'r+')
#ind2 = detect_rpeak(list(Y_true[:,0]),fs=25)
Y_true_peaks=np.zeros(len(Y_true_RA))
Y_true_peaks[ind]=1
del ind

#ind = pk.indexes(Y_pred_RA, thres=0.2, min_dist=round(Fs/(max_HR/60)))
#ind = detect_rpeak(list(Y_pred))
ind = dr_modified(list(Y_pred_RA),Fs,0.2)
ser_Y_pred=pd.Series(Y_pred_RA)
figure()
plot(ser_Y_pred);hold(True);plot(ser_Y_pred.iloc[ind],'r+')
#ind2 = detect_rpeak(list(Y_true[:,0]),fs=25)
Y_pred_peaks_RA=np.zeros(len(Y_pred_RA))
Y_pred_peaks_RA[ind]=1
del ind

ind = dr_modified(list(Y_pred_LA),Fs,0.2)
ser_Y_pred=pd.Series(Y_pred_LA)
figure()
plot(ser_Y_pred);hold(True);plot(ser_Y_pred.iloc[ind],'r+')
#ind2 = detect_rpeak(list(Y_true[:,0]),fs=25)
Y_pred_peaks_LA=np.zeros(len(Y_pred_LA))
Y_pred_peaks_LA[ind]=1
del ind

Fs=25
t_len=20 #window len in seconds
t_start=20 # prediction starting time;must be >=t_len
interv=9500//Fs #prediction interval in seconds
t_end=t_start+interv;t_ll=(t_start-t_len)*Fs+1;t_ul=t_end*Fs
windo=Fs*t_len;st=2*Fs # in seconds converted to samples
HR_Y_test=HR_predict(Y_true_peaks[t_ll:t_ul],windo,st,filt=True) #[:1500] for 1 minute
HR_DL_pred_RA=HR_predict(Y_pred_peaks_RA[t_ll:t_ul],windo,st,filt=True)
HR_DL_pred_LA=HR_predict(Y_pred_peaks_LA[t_ll:t_ul],windo,st,filt=True)
mserr_RA=round((sum((HR_Y_test-HR_DL_pred_RA)**2)/len(HR_Y_test)),3)
mserr_LA=round((sum((HR_Y_test-HR_DL_pred_LA)**2)/len(HR_Y_test)),3)
time_x=np.arange(t_start,t_end,st/Fs)
# =============================================================================
# # HR Predictions
# figure();title('Variation of Heart rate prediction with window length and step size')
# for st in range(1,11):
#     rmserr=np.zeros(100)
#     for windo in range(10,1010,10):
#         HR_Y_test=HR_predict(Y_true_peaks,windo,st)
#         HR_DL_pred=HR_predict(Y_pred_peaks,windo,st)
#         rmserr[(windo//10)-1]=(sum((HR_Y_test-HR_DL_pred)**2)/len(HR_Y_test))**0.5
#     plot(np.arange(10,1010,10),rmserr,label='step={}'.format(st));hold(True);
# legend();grid(True);xlabel('Window length');ylabel('RMSError')
# =============================================================================

figure()
plot(time_x,HR_Y_test,'b',label='True');hold(True);plot(time_x,HR_DL_pred_RA,'r--',label='Predicted_RA')
hold(True);plot(time_x,HR_DL_pred_LA,'g--',label='Predicted_LA');legend();grid(True)
title('Prediction over {} s to {} s: MSE_RA={}, MSE_LA={}'.format(t_start,\
      t_start+interv,mserr_RA,mserr_LA))