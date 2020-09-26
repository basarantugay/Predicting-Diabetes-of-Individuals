import numpy as np
import pandas as pd
from pprint import pprint
from neccessary_functions import DecisionTreeClassifierEntropy, DecisionTreeClassifierGini, calculatePerformance
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

def gini_impurity(x,y, dataset_range):
    #seperate data to training and and testing. It has been used %20 data for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = dataset_range, random_state = 1) 
    dataset_range_percentage = (1 - dataset_range) * 100
    print("\nWe are using %{} of dataset for training".format(dataset_range_percentage))
    # create gini decision tree classifier and print it 
    clf_gini = DecisionTreeClassifierGini(max_depth = 20)           #initialize decision tree classifier giny
    clf_gini.fit(x_train, y_train)                          #use training data to learn tree parameters

    giny_prediction = clf_gini.predict(x_test)[0]           #make prediction with our test data with giny DTC
    system_accuracy_giny = calculatePerformance(giny_prediction, y_test)
    print("System accuracy with giny method = {}".format(system_accuracy_giny))   #print system accuracy

    return clf_gini

def main():
    data = pd.read_csv("diabetes.csv")  #load data from csv file which we download from internet
    our_data = data.to_numpy()          #translate data to numpy array

    x = our_data[::,:8]     #get training data
    y = our_data[::,8]      #get true results
    y = y.astype(np.int64)  #convert float results to integer

    #this dataset is created for visualizing tree 
    dataset = Bunch(
        data=our_data[:, :-1],
        target=our_data[:, -1],
        feature_names=data.columns.values[:-1],
        target_names=["Diabetes", "Healthy"],
    )
    # trying different amount of data changes gini impurity, so we will try with 5 different data
    # first we seperate %70 for training and %30 for testing, then %75 for trainig and %25 for testing
    # %80 for training, and %20 for testing, %85 for training and %15 for testing and so on

    clf_gini = gini_impurity(x,y, 0.3)
    gini_impurity(x,y, 0.25)
    gini_impurity(x,y, 0.2)
    gini_impurity(x,y, 0.15)
    gini_impurity(x,y, 0.1)

    #in this method we visualize tree for gini method and save it to tree_visualization.txt
    clf_gini.debug(
        list(dataset.feature_names),
        list(dataset.target_names),
        True
    )

    #seperate data to training and and testing. It has been used %20 data for testing
    dataset_range = 0.3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = dataset_range, random_state = 1) 
    dataset_range_percentage = (1 - dataset_range) * 100 
    print("\nSystem accuracy with entropy method is calculating...")
    print("We are using %{} of dataset for training".format(dataset_range_percentage))
    clf_entropy = DecisionTreeClassifierEntropy(max_depth = 20)    #initialize decision tree classifier entropy
    m = clf_entropy.fit(x_train, y_train)                   #use training data to learn tree parameters

    entropy_prediction = clf_entropy.predict(x_test)    #make prediction with our test data with entropy DTC
    system_accuracy_entropy = calculatePerformance(entropy_prediction, y_test)
    print("System accuracy with entropy method = {}".format(system_accuracy_entropy))   #print system accuracy

    txtFile = open("tree.txt", "w")     #open txt file for saving tree parameters
    pprint(m, txtFile)                  #write it in the txt file

if __name__ == "__main__":
    main()