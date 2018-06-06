import pickle
import os
import sys
from models import CNN
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":

    path_train = "./train/"
    path_valid = "./valid/"
    obj_weight = "model_w.h5"
    obj_history = "history"

    shape = (128, 512)
    shape_t = (128, 512, 1)

    model = CNN(shape_t)

    if os.path.exists(obj_weight):
        model.load_weights(obj_weight)

    data_gen = ImageDataGenerator()
    batch_size = 32
    epochs = 16

    # samples / batch
    steps_epoch = 16
    valid_steps = 4

    train_gen = data_gen.flow_from_directory(
        path_train,
        batch_size=batch_size,
        target_size=shape,
        class_mode='categorical',
        color_mode='grayscale'
    )

    valid_gen = data_gen.flow_from_directory(
        path_valid,
        batch_size=batch_size,
        target_size=shape,
        class_mode='categorical',
        color_mode='grayscale'
    )
    history = []

    try:
        history = model.fit_generator(
            train_gen,
            steps_epoch,
            epochs,
            valid_gen,
            valid_steps
        )

        print("Saving weights")
        model.save_weights(obj_weight)

        print("Saving history")

        pickle.dump(history.history, open(obj_history, 'wb'))
    except KeyboardInterrupt:
        print("\n\n --- Interruption ---\n ---Saving weights---")
        model.save_weights(obj_weight)

        print(" ---Saving history---")
        pickle.dump(history.history, open(obj_history, 'wb'))

        sys.exit(0)


