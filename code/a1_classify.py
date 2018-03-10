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

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    print ('TODO')
    Cii_sum = 0
    Cij_sum = 0
    for i in range(4):
        Cii_sum += C[i,i]
    for i in range(4):
        for j in range(4):
            Cij_sum = C[i,j]
    return Cii_sum/Cij_sum

def calc_klass_recall(C,k):
    """Helper function for recall, that will calculate recall fraction for each class"""
    Ckj_sum = 0
    for j in range(4):
        Ckj_sum += C[k-1,j]
    return C[k-1,k-1]/Ckj_sum

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    print ('TODO')
    # the klasses
    k1 = calc_klass_recall(C,1)
    k2 = calc_klass_recall(C,2)
    k3 = calc_klass_recall(C,3)
    k4 = calc_klass_recall(C,4)

    return (k1, k2, k3, k4)

def calc_klass_precis(C,k):
    """Helper function for precision, that will calculate precision fraction for each class"""
    Cik_sum = 0
    for j in range(4):
        Cik_sum += C[i,k-1]
    return C[k-1,k-1]/Cik_sum

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    print ('TODO')
    
    k1 = calc_klass_precis(C,1)
    k2 = calc_klass_precis(C,2)
    k3 = calc_klass_precis(C,3)
    k4 = calc_klass_precis(C,4)

    return (k1, k2, k3, k4)

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')
    feats = np.load(filename)["arr_0"]     
    X = feats[:,[0,172]]                             
    y = feats[:,[173]].ravel()
    # shuffle to fix a scikit bug
    #X_shuffle, y_shuffle = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, shuffle=True, random_state=10)
    accuracies = []
    f = open('a1_3.1.csv', 'w')
    for i in range(5):
        if i == 0:
            model = "Linear" 
            clf = LinearSVC(random_state=0, max_iter=10000)
        elif i == 1:
            model = "SVC"
            clf = SVC(gamma=2, max_iter=10000)
        elif i == 2:
            model = "Forest"
            clf = RandomForestClassifier( n_estimators=10, max_depth=2)
        elif i == 3:
            model = "Neural Network"
            clf = MLPClassifier(alpha=0.05)
            
        elif i == 4:
            model = "Booster"
            clf = AdaBoostClassifier()

        clf.fit(X_train, y_train)

        X_predict = clf.predict(X_test)

        C = confusion_matrix(y_test, X_predict)

        print(C.size)

        accuracy_val = accuracy(C)
        recall_val = recall(C)
        precision_val = recall(C)

        accuracies.append(accuracy_val)
        print("done step: %d" % i)
        f.write("%d,%.5f,%.5f,,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n" % (i+1, accuracy_val,recall_val[0],recall_val[1],recall_val[2],recall_val[3],precision_val[0],precision_val[1],precision_val[2],precision_val[3], C[0,0], C[0,1], C[0,2], C[0,3], C[1,0], C[1,1], C[1,2], C[1,3], C[2,0], C[2,1], C[2,2], C[2,3], C[3,0], C[3,1], C[3,2], C[3,3]))
    f.close()

    iBest = accuracies.index(max(accuracies)) + 1

    return X_train, X_test, y_train, y_test,iBest


def bestClassifier_Return(iBest):
    # creates an instance of the best classifier and returns it as a helper function to class32
    if iBest == 1:
        model = "Linear"
        clf = LinearSVC(random_state=0, max_iter=10000)
    elif iBest == 2:
        model = "SVC"
        clf = SVC(gamma=2, max_iter=10000)
    elif iBest == 3:
        model = "Forest"
        clf = RandomForestClassifier( n_estimators=10, max_depth=2)
    elif iBest == 4:
        model = "Neural Network"
        clf = MLPClassifier(alpha=0.05)
        
    elif iBest == 5:
        model = "Booster"
        clf = AdaBoostClassifier()

    return clf

def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')
    

    clf = bestClassifier_Return(iBest)

    X_1k = X_train[0:1000]
    y_1k = y_train[0:1000]



    #~~~~training the 1k data

    clf.fit(X_1k, y_1k)

    predict_1k = clf.predict(X_test)

    C = confusion_matrix(y_test, predict_1k)
    print(C.size)
    acc_1k = accuracy(C)

    f = open("a1_3.2.csv", 'w')

    f.write("%.5f" % acc_1k)

    # amount of data we will train on for this iteration
    train_size = 5000
    for i in range(4):
        #~~~5k

        clf = bestClassifier_Return(iBest)

        X_var = X_train[0:train_size]
        y_var = y_train[0:train_size]

        clf.fit(X_var, y_var)

        predict_var = clf.predict(X_test)
        C = confusion_matrix(y_test, predict_var)
        acc_var = accuracy(C)
        train_size +=5000
        f.write(",%.5f" % acc_var)

    f.close()
    #~~~10k  
    """
    clf = bestClassifier_Return(iBest)

    X_10k = X_train[0:10000]
    y_10k = y_train[0:10000]

    predict_10k = clf.predict(X_10k)
    C = confusion_matrix(y_test, predict_10k)
    acc_10k = accuracy(C)

    #~~~15k

    clf = bestClassifier_Return(iBest)

    X_10k = X_train[0:15000]
    y_10k = y_train[0:15000]

    predict_15k = clf.predict(X_15k)
    C = confusion_matrix(y_test, predict_15k)
    acc_15k = accuracy(C)

    #~~~~20k

    clf = bestClassifier_Return(iBest)

    X_20k = X_train[0:20000]
    y_20k = y_train[0:20000]

    predict_20k = clf.predict(X_20k)
    C = confusion_matrix(y_test, predict_20k)
    acc_20k = accuracy(C)
    """




    return (X_1k, y_1k)
    
def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')

    


    p_32values = []    
    selector = SelectKBest(f_classif, 5)
    print(X_train)
    print(y_train)
    X_5_32 = selector.fit_transform(X_train, y_train)
    np.savetxt('np_out.txt', X_5_32, delimiter=',')
    p_32values.append(selector.pvalues_)

    p_1values = []    
    selector = SelectKBest(f_classif, 5)
    X_5_1 = selector.fit_transform(X_1k, y_1k)
    np.savetxt('np_out.txt', X_5_1, delimiter=',')
    p_1values.append(selector.pvalues_)

    for i in range(1,6):
        selector = SelectKBest(f_classif, i*10)
        X_curr32 = selector.fit_transform(X_train,y_train)
        p_32values.append(selector.pvalues_)

    for i in range(1,6):
        selector1 = SelectKBest(f_classif, i*10)
        X_curr1 = selector1.fit_transform(X_1k,y_1k)
        p_1values.append(selector1.pvalues_)

    f = open("a1_3.3.csv", 'w')

    f.write("5,%.5f" % p_32values[0])
    f.write("10,%.5f" % p_32values[1])
    f.write("20,%.5f" % p_32values[2])
    f.write("30,%.5f" % p_32values[3])
    f.write("40,%.5f" % p_32values[4])
    f.write("50,%.5f" % p_32values[5])

    f2 = open("3.3_other_file.txt", 'w')
    for i in range(len(p_1values)):
        f2.write("%.5f," % p_1values[i])

    f2.close()


    clf32 = bestClassifier_Return(i)
    acc32 = createCandGetAccuracy(clf32, X_5_32, y_train, X_test, y_test)
    
    clf1 = bestClassifier_Return(i)
    acc1 = createCandGetAccuracy(clf, X_5_1, y_1k, X_test, y_test)
    f.write("%.5f,%.5f" % (acc1, acc32))



    f.close()
    """
    selector = SelectKBest(f_classif, 20)
    X_20 = selector.fit_transform(X)
    p_20 = selector.fit_transform(X_train, y_train)

    selector = SelectKBest(f_classif, 30)
    X_30= selector.fit_transform(X)
    p_30 = selector.fit_transform(X_train, y_train)

    selector = SelectKBest(f_classif, 40)
    X_40= selector.fit_transform(X)
    p_40 = selector.fit_transform(X_train, y_train)

    selector = SelectKBest(f_classif, 50)
    X_50 = selector.fit_transform(X)
    p_50 = selector.fit_transform(X_train, y_train)
    """

def createCandGetAccuracy(clf, X_train, y_train, X_test, y_test):
    # creates a confusion matrix and calculates and returns accuracy

    clf.fit(X_train, y_train)

    X_predict = clf.predict(X_test)

    C = confusion_matrix(y_test, X_predict)

    accuracy_val = accuracy(C)

    return accuracy_val

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
    filename : string, the name of the npz file from Task 2
    i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')
    kf = KFold(n_splits=5, shuffle=True)

    feats = np.load(filename)["arr_0"]     
    X = feats[:,[0,172]]                             
    y = feats[:,[173]].ravel()

    clf1_results = np.zeros((5))
    clf2_results = np.zeros((5))
    clf3_results = np.zeros((5))
    clf4_results = np.zeros((5))
    clf5_results = np.zeros((5))

    j = 0
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = "Linear" 
        clf1 = LinearSVC(random_state=0, max_iter=10000)
        clf1_results = np.insert(clf1_results, j, createCandGetAccuracy(clf1, X_train, y_train, X_test, y_test) )
        
        model = "SVC"
        clf2 = SVC(gamma=2, max_iter=10000)
        clf2_results = np.insert(clf2_results, j, createCandGetAccuracy(clf2, X_train, y_train, X_test, y_test) )
        
        model = "Forest"
        clf3 = RandomForestClassifier( n_estimators=10, max_depth=2)
        clf3_results = np.insert(clf3_results, j, createCandGetAccuracy(clf3, X_train, y_train, X_test, y_test) )
        
        model = "Neural Network"
        clf4= MLPClassifier(alpha=0.05)
        clf4_results = np.insert(clf4_results, j, createCandGetAccuracy(clf4, X_train, y_train, X_test, y_test) )
        
        model = "Booster"
        clf5 = AdaBoostClassifier()
        clf5_results = np.insert(clf5_results, j, createCandGetAccuracy(clf5, X_train, y_train, X_test, y_test) )
        j+=1

    if i ==1:
        best_clf = clf1_results
    elif i ==2:
        best_clf = clf2_results
    elif i ==3:
        best_clf = clf3_results
    elif i ==4:
        best_clf= clf4_results
    elif i ==5:
        best_clf = clf5_results


    result_list = [clf1_results,clf2_results,clf3_results,clf4_results,clf5_results]


    f = open("a1_3.4.csv", 'w')

    for i in range(5):
        f.write("%.5f,%.5f,%.5f,%.5f,%.5f\n" % (result_list[0][i],result_list[1][i],result_list[2][i],result_list[3][i],result_list[4][i]))

    list_compare = []
    for array in result_list:
        if not np.array_equal(best_clf, array):
            compared = stats.ttest_rel(best_clf,array)
            list_compare.append(compared)

    f.write("%.5f,%.5f,%.5f,%.5f" % (list_compare[0][1],list_compare[1][1],list_compare[2][1],list_compare[3][1]))

    f.close()
def main(args):
    
    
    X_train, X_test, y_train, y_test,iBest = class31(args.input)

    X_1k, y_1k = class32(X_train, X_test, y_train, y_test,iBest)
    print("done 3.2")
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    print("done 3.3")
    class34(args.input, 1 )
    print("done34")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)

    args = parser.parse_args()

    main(args)

    # TODO : complete each classification experiment, in sequence.


