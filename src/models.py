import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Reshape, Permute, Bidirectional, GRU, TimeDistributed, Dense, Flatten, Activation
from keras import layers as keras_layers
from keras import backend as K
#from keras_ex.GaussianKernel import GaussianKernel, GaussianKernel2, GaussianKernel3

class EarlyConvolution:
    def __init__(self, input_layer, name='', n_filters=32, filter_length=8, strides=1, n_features=128, activation='relu'):
        self.name=name
        self.n_filters=n_filters
        self.filter_length=filter_length
        self.strides=strides
        self.n_features=n_features
        self.activation=activation
        self.padding_conv='same'
        self.input_layer=input_layer
        self.pooling=(int(0.5*self.filter_length), 1)
        self.output_layer=None
        self.trainable=True
        self.layers_df=pd.DataFrame(columns=['type', 'layer'])
        self.layers_df = self.layers_df.append({'type': 'Input' , 'layer': input_layer}, ignore_index=True)

    def define_layers(self):

        ec_conv_1 = Conv2D(filters = self.n_filters, kernel_size = self.filter_length, strides=self.strides, padding=self.padding_conv, activation=self.activation, name=self.name+'ec_conv_1', data_format='channels_last')(self.input_layer)
        ec_max_pool_1 = MaxPooling2D(pool_size=self.pooling, strides=self.pooling, name=self.name+'ec_cmax_pool_1')(ec_conv_1)

        ec_conv_2 = Conv2D(filters = self.n_filters, kernel_size = self.filter_length, strides=self.strides, padding=self.padding_conv, activation=self.activation, name=self.name+'ec_conv_2', data_format='channels_last')(ec_max_pool_1)

        ec_max_pool_2 = MaxPooling2D(pool_size=self.pooling, strides=self.pooling, name=self.name+'ec_max_pool_2')(ec_conv_2)

        self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': ec_conv_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'MaxPooling2D' , 'layer': ec_max_pool_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': ec_conv_2}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'MaxPooling2D' , 'layer': ec_max_pool_2}, ignore_index=True)

        self.output_layer=ec_max_pool_2
        return ec_max_pool_2


class LateConvolution:
    def __init__(self, input_layers, name='', n_filters=[512, 512, 6], strides=1, activation='relu', dropout=0.5):
        self.name=name
        self.n_filters=n_filters
        self.strides=strides
        self.activation=activation
        self.dropout=dropout
        self.padding_conv='valid'
        self.layers_df=pd.DataFrame(columns=['type', 'layer'])

        self.input_layers = input_layers if isinstance(input_layers, list) else [input_layers]
        self.ouput_layer=None

        for l in input_layers:
            self.layers_df = self.layers_df.append({'type': 'Input' , 'layer': l}, ignore_index=True)
            
    def define_layers(self):
        conv_input=None 
        if len(self.input_layers) > 1:
            inputs = []
            concatenate = keras_layers.concatenate(self.input_layers, axis=0)
            conv_input=concatenate
            self.layers_df = self.layers_df.append({'type': 'Concatenate' , 'layer': concatenate}, ignore_index=True)
        else:
            conv_input=self.input_layers[0]
        
        lc_conv_1 = Conv2D(filters = self.n_filters[0], kernel_size = (1, 1), strides=self.strides, padding=self.padding_conv, activation=self.activation, name=self.name+'lc_conv_1', data_format='channels_last')(conv_input)
        lc_dropout_1 = Dropout(self.dropout)(lc_conv_1) 

    
        lc_conv_2 = Conv2D(filters = self.n_filters[1], kernel_size = (1, 1), strides=self.strides, padding=self.padding_conv, activation=self.activation, name=self.name+'lc_conv_2', data_format='channels_last')(lc_dropout_1)
        lc_dropout_2 = Dropout(self.dropout)(lc_conv_2) 


        lc_conv_3 = Conv2D(filters = self.n_filters[2], kernel_size = (1, 1), strides=self.strides, padding=self.padding_conv, activation=self.activation, name=self.name+'lc_conv_3', data_format='channels_last')(lc_dropout_2)
        lc_dropout_3 = Dropout(self.dropout)(lc_conv_3) 

        self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': lc_conv_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dropout' , 'layer': lc_dropout_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': lc_conv_2}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dropout' , 'layer': lc_dropout_2}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': lc_conv_3}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dropout' , 'layer': lc_dropout_3}, ignore_index=True)

        self.output_layer=lc_dropout_3
        return lc_dropout_3

class Recurrent:
    def __init__(self, input_layer, name='', seq_len=512, rnn_dim=128, dropout=0.5, n_classes=6):
        self.name=name
        self.rnn_dim=rnn_dim
        self.seq_len=seq_len
        self.dropout=dropout
        self.n_classes=n_classes

        self.layers_df=pd.DataFrame(columns=['type', 'layer'])

        self.input_layer=input_layer
        self.ouput_layer=None

    def define_layers(self):
#        re_permute = Permute((2, 1, 3), name='permute')(self.input_layer)
        re_reshape = Reshape((self.seq_len, -1), name='reshape')(self.input_layer)
        re_gru_1 = Bidirectional(
            GRU(self.rnn_dim, activation='tanh', dropout=self.dropout, recurrent_dropout=self.dropout, return_sequences=True, name='gru_1'), merge_mode='mul'
        )(re_reshape)
        re_gru_2 = Bidirectional(
            GRU(self.rnn_dim, activation='tanh', dropout=self.dropout, recurrent_dropout=self.dropout, return_sequences=True, name='gru_2'), merge_mode='mul'
        )(re_gru_1)
        
        re_dense_1 = TimeDistributed(Dense(self.n_classes, name='re_dense_1'))(re_gru_1)
        re_relu_l= Activation('relu', name='re_relu_l')(re_dense_1)
        re_drop_l = Dropout(self.dropout)(re_relu_l)

        re_dense_2 = TimeDistributed(Dense(self.n_classes, name='re_dense_2'))(re_drop_l)
        re_sigmoid_l = Activation('sigmoid', name='re_sigmoid_l')(re_dense_2)

#        self.layers_df = self.layers_df.append({'type': 'Permute' , 'layer': re_permute}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Reshape' , 'layer': re_reshape}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'BiGru' , 'layer': re_gru_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'BiGru' , 'layer': re_gru_2}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dense' , 'layer': re_dense_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Activation' , 'layer': re_relu_l}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dropout' , 'layer': re_drop_l}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dense' , 'layer': re_dense_2}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Activation' , 'layer': re_sigmoid_l}, ignore_index=True)

        self.output_layer = re_sigmoid_l

        return re_sigmoid_l
  
class Classifier:
    def __init__(self, input_layer, name='', dense_dim=32, dropout=0.5, n_classes=6):
        self.name=name
        self.dense_dim=dense_dim
        self.droput=dropout
        self.n_classes=n_classes

        self.layers_df=pd.DataFrame(columns=['type', 'layer'])

        self.input_layer=input_layer
        self.ouput_layer=None

    def define_layers(self):

        cl_dense_1 = TimeDistributed(Dense(1))(self.input_layer)
        cl_flatten = Flatten()(cl_dense_1)
        cl_dense_2 = Dense(self.dense_dim)(cl_flatten)
        cl_dense_3 = Dense(self.n_classes)(cl_dense_2)
        cl_sigmoid_l = Activation('sigmoid', name='cl_sigmoid_l')(cl_dense_3)

        self.layers_df = self.layers_df.append({'type': 'Dense' , 'layer': cl_dense_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Flatten' , 'layer': cl_flatten}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dense' , 'layer': cl_dense_2}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Dense' , 'layer': cl_dense_3}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Activation' , 'layer': cl_sigmoid_l}, ignore_index=True)
        
        self.output_layer = cl_sigmoid_l
        return cl_sigmoid_l
        


