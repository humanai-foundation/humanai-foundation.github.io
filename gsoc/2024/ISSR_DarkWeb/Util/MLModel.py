import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import pickle
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def plot_loss(eval_result: dict, title: str = "") -> None:
    """
    Plot the training and validation loss
    :param eval_result: The evaluation results
    """
    train_losses = eval_result['training']['multi_logloss']
    valid_losses = eval_result['valid_1']['multi_logloss']
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.xlabel('Boosting round')
    plt.ylabel('Multi Log Loss')
    plt.title(f'Training and Validation Loss {title}')
    plt.legend();

def k_fold(X: pd.DataFrame, y: pd.Series, params: dict, n_splits: int = 5) -> dict:
    """
    Perform k-fold cross-validation on the training set.
    :param X: the training features
    :param y: the training labels
    :param params: the parameters for the LightGBM model
    :param n_splits: number of splits
    :return: classification report for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    class_accuracies = {i: [] for i in range(len(y.unique()))}

    fold = 1

    # Loop through the folds
    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        eval_result = {}

        # Train the model
        bst_fold = lgb.train(params,
                        train_data,
                        valid_sets=[train_data, valid_data],
                        num_boost_round=1000,
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=3),
                            lgb.record_evaluation(eval_result)
                        ])

        # Predictions on the validation set
        y_valid_pred = np.argmax(bst_fold.predict(X_valid), axis=1)

        # Calculate classification report
        report = classification_report(y_valid, y_valid_pred, output_dict=True)

        # Extract per-class accuracy (recall)
        for class_id in class_accuracies.keys():
            class_accuracies[class_id].append(report[str(class_id)]['recall'])

        plot_loss(eval_result, title=f"Fold {fold}")

        fold += 1

    return class_accuracies

def summarize_k_fold(class_accuracies: dict) -> pd.DataFrame:
    """
    Summarize the results of the k-fold cross-validation
    :param class_accuracies: the results of the k-fold cross-validation
    :return: a DataFrame with the mean and standard deviation of the per-class accuracies
    """
    class_accuracies = pd.DataFrame(class_accuracies)
    class_accuracies['mean'] = class_accuracies.mean(axis=1)
    class_accuracies['std'] = class_accuracies.std(axis=1)
    return class_accuracies

def save_model(output_file: str, bst: lgb.Booster) -> None:
    """
    Save the trained model to a file.
    :param output_file: the output file
    :param bst: the trained model
    """
    with open(output_file, 'wb') as f:
        pickle.dump(bst, f)