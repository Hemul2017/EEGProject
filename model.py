import pandas as pd
from pathlib import Path
import os
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

base_path = Path('./epochs_tensors_w_connectivity/imaginary')
data = []
for filename in os.listdir(base_path)[:-1]:
    tensor = torch.load(base_path / filename)
    data.append(tensor.flatten()[::10])
    #data.append(np.abs(np.fft.rfft(tensor.flatten()[::10])))


X = np.array(data)
y = np.array(torch.load(base_path / 'labels.pt'))
print(X.shape)
print(y.shape)

skf = StratifiedKFold(shuffle=True)
rf_model = RandomForestClassifier(class_weight='balanced_subsample')
lda_model = LinearDiscriminantAnalysis()

mean_acc = []
mean_f1 = []
for train_index, test_index in skf.split(X, y):
    lda_model.fit(X[train_index], y[train_index])
    y_pred = lda_model.predict(X[test_index])
    acc = accuracy_score(y[test_index], y_pred)
    f1 = f1_score(y[test_index], y_pred, average='weighted')
    mean_acc.append(acc)
    mean_f1.append(f1)
    print(acc)
    print(f1)

print(mean_acc)
print(np.mean(mean_acc))
print(mean_f1)
print(np.mean(mean_f1))





