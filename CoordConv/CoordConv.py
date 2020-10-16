from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *

class CoordNet(Network):    
    def __init__(self, input_shape):#assume coordlayers are added before  
        super().__init__()

        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = 1
        self.lr = 0.0001

        optimizer = keras.optimizers.Adam(lr=self.lr)

        self.inp = layers.Input(input_shape)
        self.Net = tf.keras.models.Model(self.inp, self.tf_resnet(self.inp))
        self.Net.compile(loss=tf.keras.losses.MSE,\
            optimizer=optimizer)
        self.es = tf.keras.callbacks.EarlyStopping(monitor='loss',\
            patience=10, verbose=2, restore_best_weights=True)

    def tf_resnet(self,x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainable = True
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256,activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128,activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1)(x)
        return x

    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):
        batch_hist = LossHistory()

        History = self.Net.fit(x_train, y_train,\
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(x_test, y_test),
            callbacks=[batch_hist])

        epochs_arr = np.arange(self.curr_epoch)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        self.hist['train_MSE'].append(History.history['loss'])
        self.hist['batch_MSE'].append(batch_hist.history['loss'])
        self.hist['test_MSE'].append(History.history['val_loss'])
