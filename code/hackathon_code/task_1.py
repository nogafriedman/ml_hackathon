import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .preprocess import get_preprocessed_data


def learn_task_1(train_df):
    X_train, y_train = train_df.drop("did_cancel", axis=1), train_df["did_cancel"]

    # Unfortunately we needed to lower the max_depth from 25
    model = RandomForestClassifier(max_depth=16)
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'task_1.joblib')


def predict_task_1(model, test_df, train_df):
    ids = test_df["h_booking_id"]
    clean_test = get_preprocessed_data(test_df, task=1, is_test=True).reindex(train_df.columns, axis=1, fill_value=0)
    clean_test.drop("did_cancel", axis=1, inplace=True)

    y_pred = model.predict(clean_test)
    result_df = pd.DataFrame({"id": ids, "cancellation": y_pred})
    # result_df.to_csv("../predictions/agoda_cancellation_prediction.csv", index=False)
    return result_df
