import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

from task_1.code.hackathon_code.preprocess import get_preprocessed_data


def decision_tree_selection(X_train, y_train, depth_values):
    scores = []

    for depth in depth_values:
        model = DecisionTreeClassifier(max_depth=depth)
        scores.append(cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro").mean())

    plt.plot(depth_values, scores, marker='o')
    plt.xlabel('Maximum Depth')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Maximum Depth - Decision Tree')
    plt.show()


def random_forest_selection(X_train, y_train, depth_values):
    scores = []

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    for depth in depth_values:
        model = RandomForestClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        scores.append(f1_score(y_test, model.predict(X_test), average="macro"))

    plt.plot(depth_values, scores, marker='o')
    plt.xlabel('Maximum Depth')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Maximum Depth - Random Forest')
    plt.show()


def regression_selection(X_train, y_train, alpha_values):
    scores = []

    for alpha in alpha_values:
        model = HistGradientBoostingRegressor(max_depth=alpha, l2_regularization=0.1, validation_fraction=None)
        scores.append(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean())

    plt.plot(alpha_values, scores, marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('-RMSE')
    plt.title('-RMSE vs Max Depth - HistGradientBoostingRegression (ridge=0.1)')
    plt.show()


def classification_feature_selection(X_train, y_train, max_features):
    scores = []

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train_scaled, y_train)
    feature_names = list(X_train.columns)
    coefficients = lasso.coef_

    sorted_features = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    sorted_feature_names = [name for name, _ in sorted_features]

    for i in range(1, max_features + 1, 2):
        new_X_train = X_train[sorted_feature_names[:i]]
        new_X_test = X_test.reindex(new_X_train.columns, axis=1, fill_value=0)
        model = RandomForestClassifier(max_depth=24)
        model.fit(new_X_train, y_train)
        scores.append(f1_score(y_true=y_test, y_pred=model.predict(new_X_test), average="macro"))

    plt.plot(list(range(1, max_features + 1, 2)), scores, marker='o')
    plt.xlabel('Max Features')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Max Features - RandomForestClassifier (depth=24)')
    plt.show()


def regression_feature_selection(X_train, y_train, max_features):
    scores = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_scaled, y_train)
    feature_names = list(X_train.columns)
    coefficients = lasso.coef_

    sorted_features = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    sorted_feature_names = [name for name, _ in sorted_features]

    for i in range(1, max_features + 1, 2):
        new_X_train = X_train[sorted_feature_names[:i]]
        model = HistGradientBoostingRegressor(validation_fraction=None)
        scores.append(cross_val_score(model, new_X_train, y_train, cv=5, scoring='neg_root_mean_squared_error').mean())

    plt.plot(list(range(1, max_features + 1, 2)), scores, marker='o')
    plt.xlabel('Max Features')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Max Features - RandomForestClassifier (depth=24)')
    plt.show()


def correlation(df):
    # Feature for correlation comparison
    target = 'did_cancel'

    # Compute the correlation coefficients with the selected feature
    correlation = df.corr()[target].drop(target)

    # Correlation threshold
    threshold = 0.1

    # Filter features based on the correlation threshold
    correlation_filtered = correlation[abs(correlation) >= threshold]

    # Plot of the correlation coefficients
    plt.figure(figsize=(5, 5))
    correlation_filtered.plot(kind='bar')
    plt.title(f'Correlation with {target} (Threshold: {threshold})')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()


def plot_loss_as_a_function_of_policy(df):
    df = df.sample(n=1000, random_state=42)

    # Leave only cancellations data
    df['cancellation_datetime'].dropna()

    # Calculate loss
    df['Loss'] = df['original_selling_amount']

    # Threshold for the loss feature
    threshold = 2000

    # Filter the data based on the threshold
    df = df[df['Loss'] > threshold]

    # Sort dataframe by loss in descending order
    df = df.sort_values('Loss', ascending=True)

    # Create bar plot
    plt.figure(figsize=(15, 9))
    plt.bar(df['cancellation_policy_code'], df['Loss'])
    plt.title('Loss as a Function of Policy')
    plt.xlabel('Policy')
    plt.ylabel('Loss')
    plt.xticks(rotation=0)

    # Display the plot
    plt.show()


def plot_some_graphs(train_df):
    train_df_1 = get_preprocessed_data(train_df, task=1)
    X_train, y_train = train_df_1.drop("did_cancel", axis=1), train_df_1["did_cancel"]
    # decision_tree_selection(X_train, y_train, [i for i in range(1, 19, 2)])
    random_forest_selection(X_train, y_train, [i for i in range(10, 21, 2)])
    correlation(train_df_1)
    # classification_feature_selection(X_train, y_train, len(X_train.columns))

    train_df_2 = get_preprocessed_data(train_df, task=2)
    X_train, y_train = train_df_2.drop("original_selling_amount", axis=1), train_df_2["original_selling_amount"]
    regression_selection(X_train, y_train, alpha_values=list(range(1, 20, 2)))
    plot_loss_as_a_function_of_policy(train_df)
    # regression_feature_selection(X_train, y_train, len(X_train.columns))
