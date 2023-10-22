import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .preprocess import get_preprocessed_data


def learn_task_2(train_df):
    X_train, y_train = train_df.drop("original_selling_amount", axis=1), train_df["original_selling_amount"]

    model = HistGradientBoostingRegressor(validation_fraction=None)
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'task_2.joblib')


def predict_task_2(model, test_df, train_df, cls_train_df, class_model):
    ids = test_df["h_booking_id"]

    clean_test = get_preprocessed_data(test_df, task=2, is_test=True).reindex(train_df.columns, axis=1, fill_value=0)
    clean_test_cls = get_preprocessed_data(test_df, task=1, is_test=True).reindex(cls_train_df.columns, axis=1,
                                                                                  fill_value=0)

    clean_test_cls.drop("did_cancel", axis=1, inplace=True)
    clean_test.drop("original_selling_amount", axis=1, inplace=True)

    y_pred = model.predict(clean_test)
    cancellations = class_model.predict(clean_test_cls)

    y_pred[cancellations == 0] = -1

    result_df = pd.DataFrame({"id": ids, "predicted_selling_amount": y_pred})
    # result_df.to_csv("../predictions/agoda_cost_of_cancellation.csv", index=False)
    return result_df
