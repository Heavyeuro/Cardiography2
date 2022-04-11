from collections import OrderedDict
from random import uniform

from matplotlib import pyplot
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import core_action as ca
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# beside_list - columns to avoid
def build_and_score_ml_model_core(X_full: DataFrame, beside_list=[]):
    target_column = 'diagnostic_superclass'
    randomState = round(uniform(0, 300))
    y = X_full[target_column]
    X = ca.null_to_NaN(X_full.drop([target_column], axis=1), beside_list)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.82, test_size=0.18, random_state=randomState)

    estimate(X_train, X_valid, y_train, y_valid, randomState)
    rfc(X_train, X_valid, y_train, y_valid, randomState)
    # rfc_(X_train, X_valid, y_train, y_valid)


def estimate(X_train, X_valid, y_train, y_valid, random_state=round(uniform(0, 300))):
    names = [
        "Nearest Neighbors",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(random_state=random_state, min_samples_split=20, max_features='log2', criterion='entropy'),
        RandomForestClassifier(random_state=random_state, min_samples_split=20, max_features='log2', criterion='entropy'),
        MLPClassifier(alpha=1, max_iter=1000)
    ]

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_valid)
        predictions = [round(value) for value in predictions]
        accuracy = accuracy_score(y_valid, predictions)
        print(name +" :Accuracy: %.3f%%" % (accuracy * 100.0))
        print(name + " :Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))


def rfc(X_train, X_valid, y_train, y_valid, randomState=round(uniform(0, 300))):
    reg_model = RandomForestClassifier(n_estimators=10000, random_state=randomState, n_jobs=-1)
    reg_model = reg_model.fit(X_train, y_train)
    predicted = reg_model.predict(X_valid)
    predictions = [round(value) for value in predicted]
    accuracy = accuracy_score(y_valid, predictions)
    build_confusion_matrix(predicted, y_valid)
    print("Accuracy: %.3f%%" % (accuracy * 100.0))
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))


def rfc_(X_train, X_valid, y_train, y_valid):
    RANDOM_STATE = round(uniform(0, 100))
    ensemble_clfs = [
        (
            "RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs = -1
            ),
        ),
        (
            "RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                warm_start=True,
                max_features="log2",
                oob_score=True,
                random_state=RANDOM_STATE,
                n_jobs = -1
            ),
        ),
        (
            "RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE,
                n_jobs = -1
            ),
        ),
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 2
    max_estimators = 1000

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train, )

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def performance_measurement(my_model):
    # retrieve performance metrics
    results = my_model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    pyplot.ylabel('Log Loss')
    pyplot.title('XGBoost Log Loss')
    pyplot.show()
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()


def build_confusion_matrix(predictions, y_test):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()



