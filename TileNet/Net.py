import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as layers
from tensorflow.keras import backend as K

from sklearn.utils import shuffle

class TileLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TileLayer, self).__init__()
        self.L = 128
        self.f = 32
        self.n = int(self.L/self.f)
        self.dims = 5

    def call(self, x):
        tiles = tf.image.extract_patches(x,\
            sizes=[1,self.f,self.f,1], strides=[1,self.f,self.f,1],\
            rates=[1,1,1,1], padding='SAME')

        tiles = tf.reshape(tiles, [-1, self.n**2, self.f, self.f, self.dims])
        #tiles = tf.reshape(tiles, [-1, 16,self.f,self.f,self.dims])
        return tiles

Tiler = TileLayer()

class Network():
    def __init__(self, input_shape):
        
        self.dirpath = 'records'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

         
        self.batch_size = 64
        self.input_shape = input_shape
        self.L = self.input_shape[0]
        self.dims = self.input_shape[-1]
        self.f = 32
        self.tile_shape = [self.f, self.f, self.dims]
        #self.tile_shape = [x/self.f for x in self.input_shape]       
        self.reset()

        self.img_inp = layers.Input(self.input_shape)
        self.tile_inp = layers.Input(self.tile_shape)
        self.tile_Net = tf.keras.models.Model(self.tile_inp, self.tf_resnet(self.tile_inp))
        self.Net = tf.keras.models.Model(self.img_inp, self.thismodel(self.img_inp, self.tile_Net))
        
        optimizer = keras.optimizers.Adam(lr=0.001)
        self.Net.compile(loss=tf.keras.losses.MSE,\
            optimizer=optimizer)
   
    def get_tiles(self,x):
        tiles = tf.image.extract_patches(x,\
            sizes=[1,self.f,self.f,1], strides=[1,self.f,self.f,1],\
            rates=[1,1,1,1], padding='SAME')
        tiles = np.reshape(tiles, [tiles.shape[0], tiles.shape[1]**2, self.f, self.f, self.dims])
        return tiles
   
    def testnet(self,x):
        x = layers.Conv2D(4,4)(x)
        x = layers.Flatten()(x)

        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(1)(x)
        return x

    def tf_resnet(self,x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.tile_shape)
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(1)(x)
        return x

    def thismodel(self,x,net):
        tile_layer = Tiler(x)
        tilesT = tf.transpose(tile_layer,[1,0,2,3,4])
       
        predT = [net(patch) for patch in tilesT]
        pred = tf.transpose(predT,[1,0,2])

        pred = tf.reduce_max(pred,axis=1)
        pred = K.flatten(pred)
        return pred
        
    def reset(self):
        self.curr_epoch = 0
        self.hist = defaultdict(list)

    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):
        
        History = self.Net.fit(x_train, y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(x_test, y_test))
        
        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        self.hist['train_MSE'].append(History.history['loss'])
        self.hist['test_MSE'].append(History.history['val_loss'])
    
        self.curr_epoch += epochs

    def predict(self, x_test):
        preds = self.Net.predict(x_test,\
            batch_size=self.batch_size,
            verbose=0)      
        return preds

    def evaluate(self, x_test, y_test):
        return self.Net.evaluate(x_test, y_test)

    def save(self, path):
        f = open(self.dirpath + path + '.pickle', 'wb')
        pickle.dump(self.hist,f)
        f.close()
        self.Net.save_weights(self.dirpath + path + '.h5')
    
    def load(self, path):
        self.reset()
        f = open(self.dirpath + path + '.pickle', 'rb')
        self.hist = pickle.load(f)
        f.close()
        self.Net.load_weights(self.dirpath + path + '.h5')
        self.curr_epoch = self.hist['epochs'][-1][-1]

