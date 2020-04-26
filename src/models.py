import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling1D, Concatenate, Dropout

class EarlyConvolution:
    def __init__(self, input_layer, n_filters=32, filter_length=8, strides=1, n_features=128, activation='relu'):
        self.n_filters=n_filters
        self.filter_length=filter_length
        self.strides=strides
        self.n_features=n_features
        self.activation=activation
        self.padding_conv='same'
        self.input_layer=input_layer
        self.output_layer=None
        self.layers_df=pd.DataFrame(columns=['type', 'layer'])
        self.layers_df = self.layers_df.append({'type': 'input' , 'layer': input_layer}, ignore_index=True)

    def define_layers(self):

        early_conv_1 = Conv2D(filters = self.n_filters, kernel_size = (self.filter_length, self.n_features), strides=self.strides, padding=self.padding_conv, activation=self.activation, name='early_conv_1')(self.input_layer)

        max_pool_1 = MaxPooling2D(pool_size=(filter_length/2, 1), strides=(filter_length/2, 1), name='max_pool_1')(early_conv_1)

        early_conv_2 = Conv2D(filters = self.n_filters, kernel_size = (self.filter_length, 1), strides=self.strides, padding=self.padding_conv, activation=self.activation, name='early_conv_2')(max_pool_1)

        max_pool_2 = MaxPooling2D(pool_size=(filter_length/2, 1), strides=(filter_length/2, 1), name='max_pool_2')(early_conv_2)

        self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': early_conv_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'MaxPooling2D' , 'layer': max_pool_1}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': early_conv_2}, ignore_index=True)
        self.layers_df = self.layers_df.append({'type': 'MaxPooling2D' , 'layer': max_pool_2}, ignore_index=True)

        self.output_layer=max_pool_2
        return max_pool_2

class LateConvolution:
    def __init__(self, input_layers, n_filters=[512, 512, 6], strides=1, activation='relu', dropout=0.5):
        self.n_filters=n_filters
        self.strides=strides
        self.activation=activation
        self.dropout=dropout
        self.padding_conv='valid'
        self.layers_df=pd.DataFrame(columns=['type', 'layer'])

        self.input_layers = input_layers if isinstance(input_layers, list) else [input_layers]
        self.ouput_layer=None

        for l in input_layers:
            self.layers_df = self.layers_df.append({'type': 'input' , 'layer': l}, ignore_index=True)
            
        def define_layers(self):
            conv_input=None 
            if len(self.input_layers) > 1:
                concatenate = Concatenate(self.input_layers, axis=1)
                conv_input=concatenate
                self.layers_df = self.layers_df.append({'type': 'Concatenate' , 'layer': concatenate}, ignore_index=True)
            else:
                conv_input=self.input_layers[0]
            
            late_conv_1 = Conv2D(filters = self.n_filters[0], kernel_size = (1, 1), strides=self.strides, padding=self.padding_conv, activation=self.activation, name='late_conv_1')(conv_input)
            dropout_1 = Dropout(self.dropout)(late_conv_1) 

        
            late_conv_2 = Conv2D(filters = self.n_filters[1], kernel_size = (1, 1), strides=self.strides, padding=self.padding_conv, activation=self.activation, name='late_conv_2')(dropout_1)
            dropout_2 = Dropout(self.dropout)(late_conv_2) 


            late_conv_3 = Conv2D(filters = self.n_filters[2], kernel_size = (1, 1), strides=self.strides, padding=self.padding_conv, activation=self.activation, name='late_conv_3')(dropout_2)
            dropout_3 = Dropout(self.dropout)(late_conv_3) 

            self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': late_conv_1}, ignore_index=True)
            self.layers_df = self.layers_df.append({'type': 'Dropout' , 'layer': dropout_1}, ignore_index=True)
            self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': late_conv_2}, ignore_index=True)
            self.layers_df = self.layers_df.append({'type': 'Dropout' , 'layer': dropout_2}, ignore_index=True)
            self.layers_df = self.layers_df.append({'type': 'Conv2D' , 'layer': late_conv_3}, ignore_index=True)
            self.layers_df = self.layers_df.append({'type': 'Dropout' , 'layer': dropout_3}, ignore_index=True)

            self.output_layer=dropout_3
            return dropout_3
