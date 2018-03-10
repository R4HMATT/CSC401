from a1_classify import *
from sklearn.model_selection import train_test_split, KFold
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os


def main():
    X_train, X_test, y_train, y_test,iBest = class31("feats.npz")

    X_1k, y_1k = class32(X_train, X_test, y_train, y_test,iBest)

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

if __name__ == "__main__":
    main()