import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.keras.models import load_model

def main():
    # load model
    path_model = 'model201.keras'
    model = load_model(path_model)

    # load score
    score_one_hot = open("../SR-ARE-score/names_onehots.pickle", "rb")
    scoretent = pickle.load(score_one_hot)
    x_score = scoretent["onehots"]

    # predict the output
    pred_score = model.predict(x=x_score)

    # generate the output file
    checking = pred_score >= 0.5
    answer = checking.astype(int)
    f = open("labels.txt", "w")
    for i in range(len(answer)):
        f.write(str(answer[i][0]) + "\n")
    f.close()

if __name__=='__main__':
    main()