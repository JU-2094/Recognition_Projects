import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input, Dropout, Dense


class CNN:

    def __init__(self, shape):
        self.model = self._create_model(shape)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.adam(),
                           metrics=[self.metric_auc_roc_score, 'accuracy'])
        print(self.model.summary())

    @staticmethod
    def _create_model(shape):
        x = Input(shape)
        h = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)
        h = Dropout(0.5)(h)
        h = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=(2, 4))(h)
        h = Dropout(0.5)(h)
        h = Flatten()(h)
        o = Dense(units=3, activation='softmax')(h)
        return Model(inputs=x, outputs=o)

    def fit_generator(self, gen, steps_per_epoch, epochs, valid_data, valid_steps):
        return self.model.fit_generator(generator=gen,
                                        steps_per_epoch=steps_per_epoch,
                                        epochs=epochs,
                                        validation_data=valid_data,
                                        validation_steps=valid_steps,
                                        verbose=1)

    def evaluate_generator(self, gen):
        return self.model.evaluate_generator(gen)

    def save_weights(self, path):
        self.model.save(path)

    def load_weights(self, path):
        self.model.load_weights(path, False)

    @staticmethod
    def metric_auc_roc_score(y_true, y_pred):
        """
               Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
               from prediction scores.
        """

        value, update_op = tf.metrics.auc(y_true, y_pred)

        metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value



