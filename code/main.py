import joblib
import pandas as pd

from hackathon_code.task_1 import learn_task_1, predict_task_1
from hackathon_code.task_2 import learn_task_2, predict_task_2
from task_1.code.hackathon_code.model_selection import plot_some_graphs
from task_1.code.hackathon_code.preprocess import get_preprocessed_data

if __name__ == '__main__':
    train = pd.read_csv("../datasets/agoda_cancellation_train.csv")
    clean_train1 = get_preprocessed_data(train, task=1)
    clean_train2 = get_preprocessed_data(train, task=2)
    # Uncomment as needed
    # learn_task_1(clean_train1)
    # learn_task_2(clean_train2)

    task1_test = pd.read_csv("../datasets/Agoda_Test_1.csv")
    task2_test = pd.read_csv("../datasets/Agoda_Test_2.csv")

    try:
        task_1_model = joblib.load('task_1.joblib')
        prediction_1 = predict_task_1(task_1_model, task1_test, clean_train1)
    except:
        pass

    try:
        task_1_model = joblib.load('task_1.joblib')
        task_2_model = joblib.load('task_2.joblib')
        prediction_2 = predict_task_2(task_2_model, task2_test, clean_train2, clean_train1, task_1_model)
    except:
        pass

    # Uncomment if needed
    # plot_some_graphs(train)
