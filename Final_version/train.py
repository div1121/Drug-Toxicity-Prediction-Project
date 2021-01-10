import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model

def finddrugdict():
    # input for labels
    train_labels = open("SR-ARE-train/names_labels.txt", "r")
    content = train_labels.read()
    content_list = re.split(",|\n", content)
    drug_dict = {content_list[2 * i]: float(content_list[2 * i + 1]) for i in range(len(content_list) // 2)}
    return drug_dict

def finddrugnamedict():
    # input for similes
    train_smiles = open("SR-ARE-train/names_smiles.txt", "r")
    dtent = train_smiles.read()
    dtent_list = re.split(",|\n", dtent)
    drug_name_dict = {dtent_list[2 * i]: dtent_list[2 * i + 1] for i in range(len(dtent_list) // 2)}
    return drug_name_dict

def findtrainpickle():
    # input for data
    train_one_hot = open("SR-ARE-train/names_onehots.pickle", "rb")
    rtent = pickle.load(train_one_hot)
    return rtent

def findtestlabel():
    test_labels = open("SR-ARE-test/names_labels.txt", "r")
    ttcontent = test_labels.read()
    ttcontent_list = re.split(",|\n", ttcontent)
    ttdrug_dict = {ttcontent_list[2 * i]: float(ttcontent_list[2 * i + 1]) for i in range(len(ttcontent_list) // 2)}
    return ttdrug_dict

def findtestnamedict():
    test_smiles = open("SR-ARE-test/names_smiles.txt", "r")
    ttdtent = test_smiles.read()
    ttdtent_list = re.split(",|\n", ttdtent)
    ttdrug_name_dict = {ttdtent_list[2 * i]: ttdtent_list[2 * i + 1] for i in range(len(ttdtent_list) // 2)}
    return ttdrug_name_dict

def findtestpickle():
    # input for test data
    test_one_hot = open("SR-ARE-test/names_onehots.pickle", "rb")
    ttrtent = pickle.load(test_one_hot)
    return ttrtent

def createmodel(drug_height,drug_width,drug_full_shape,no_class):
    # build model
    model = Sequential()
    model.add(InputLayer(input_shape=(drug_height, drug_width,)))

    model.add(Reshape(drug_full_shape))

    model.add(Conv2D(kernel_size=(1, 16), strides=1, filters=8, padding='same', activation='relu', name="conv1"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same'))

    model.add(Conv2D(kernel_size=(1, 16), strides=1, filters=16, padding='same', activation='relu', name="conv2"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same'))

    model.add(Conv2D(kernel_size=(1, 16), strides=1, filters=32, padding='same', activation='relu', name="conv3"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same'))

    model.add(Flatten())

    model.add(Dense(no_class, activation="sigmoid", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

    return model

def main():
    # input files
    drug_dict = finddrugdict()
    drug_name_dict = finddrugnamedict()
    rtent = findtrainpickle()
    ttdrug_dict = findtestlabel()
    ttdrug_name_dict = findtestnamedict()
    ttrtent = findtestpickle()

    # extract training data
    x_train = rtent["onehots"]
    label = rtent["names"]
    zlabel = [drug_dict[label[i]] for i in range(len(label))]
    y_train = np.array(zlabel, float)

    # extract test data
    x_test = ttrtent["onehots"]
    ttlabel = ttrtent["names"]
    ttzlabel = [ttdrug_dict[ttlabel[i]] for i in range(len(ttlabel))]
    y_test = np.array(ttzlabel, float)

    # define training data parameter
    drug_height = x_train.shape[1]
    drug_width = x_train.shape[2]
    drug_shape = (drug_height, drug_width)
    drug_full_shape = (drug_height, drug_width, 1)
    no_class = 1
    epochs = 26
    correct = y_train >= 0.5
    true_label = correct[correct].astype(int)
    incorrect = y_train <= 0.5
    false_label = incorrect[incorrect].astype(int)
    weight_for_zero = 1.0 / len(false_label)
    weight_for_one = 1.0 / len(true_label)
    # print(weight_for_zero, weight_for_one)

    # create model
    model = createmodel(drug_height,drug_width,drug_full_shape,no_class)
    print(model.summary())
    optimizer = Adam(lr=1e-6)
    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
    class_weight = {0: weight_for_zero, 1: weight_for_one}
    model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=128, class_weight=class_weight, validation_data=(x_test, y_test))

    # evaluate model
    result = model.evaluate(x=x_test, y=y_test)
    result2 = model.evaluate(x=x_train, y=y_train)

    # save model
    # path_model = 'model.keras'
    # model.save(path_model)
    
if __name__=='__main__':
    main()