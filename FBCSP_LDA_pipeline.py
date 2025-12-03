


from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from DataLoader2 import DataLoader2
from FBCSP import FBCSP
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_data():
    data_path = Path('./working_data_inter/')
    labels_path = data_path / 'labels.pt'
    data_loader = DataLoader2(data_path, labels_path)
    pattern = r'^(i|r)_\d{3}_\d{2}_\d{2}_\d.pt$'
    X, y = data_loader.load(pattern=pattern)
    X = X.astype(np.float64)

    return X, y


def main():
    random_state = 17
    X, y = load_data()


    # Initialise models
    svm = SVC(class_weight='balanced', random_state=random_state)
    lda = LinearDiscriminantAnalysis()
    fbcsp = FBCSP(n_csp_components=4, transform_into='average_power')

    skf = StratifiedKFold(shuffle=True, random_state=random_state)

    mean_acc = []
    mean_f1 = []
    zero_fractions = []
    for train_index, test_index in skf.split(X, y):

        # Split data
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Transform train data
        fbcsp.fit(X_train, y_train, [(1, 4), (4, 8), (8, 12), (12, 18), (18, 30)])
        X_train = fbcsp.transform(X_train, reshape=True)

        # Fit model
        #lda.fit(X_train, y_train)
        svm.fit(X_train, y_train)

        # Transform test data
        X_test = fbcsp.transform(X_test, reshape=True)

        # Predict y_test
        #y_pred = lda.predict(X_test)
        y_pred = svm.predict(X_test)

        zero_fractions.append(np.mean(y_pred == 0))

        # Estimation
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mean_acc.append(acc)
        mean_f1.append(f1)
        plt.show()

    print(mean_acc)
    print(np.mean(mean_acc))
    print(mean_f1)
    print(np.mean(mean_f1))
    print(f'{np.mean(zero_fractions) * 100}% of zeros')
    print(y_test)
    print(y_pred)


if __name__ == '__main__':
    main()