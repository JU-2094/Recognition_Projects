import os
import pickle
from models import CNN
from keras.preprocessing.image import ImageDataGenerator

if __name__ == "__main__":

    path_test = "./test/"

    obj_weight = "model_w.h5"

    shape = (128, 512)
    shape_t = (128, 512, 1)

    model = CNN(shape_t)
    if not os.path.exists(obj_weight):
        raise ValueError("Can't find model weights")

    model.load_weights(obj_weight)

    data_gen = ImageDataGenerator()

    batch_size = 2

    test_gen = data_gen.flow_from_directory(
        path_test,
        batch_size=batch_size,
        target_size=shape,
        class_mode='categorical',
        color_mode='grayscale'
    )

    results = model.evaluate_generator(test_gen)

    print("Results:: \n\n", results)
    # loss, roc_auc, accuracy
    # [3.939978941281644, 0.7603900988896688]
    pickle.dump(results, open("results", 'wb'))



