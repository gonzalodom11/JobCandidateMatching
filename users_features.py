from collections import defaultdict
import json
import graphviz
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

#from data_representation import get_dict_matches

#interactions1235 = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\data\interactions1235.csv")
#train_set = pd.read_csv(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\train.csv")
#interactions1235 = interactions1235.drop('timestamp', axis=1)

#print(max(interactions1235[interactions1235["user"]==4].get("interaction").tolist()))

def read_and_parse_data(file_path):
    data_dict = defaultdict(list)
    new_file = ""
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            index = int(parts[0])
            for p in parts[1:]:
                data_dict[index].append(int(p))
    return data_dict

def read_dict(file):
    matches_dict = defaultdict(int)
    with open(file, "r") as file:
        next(file)
        for line in file:
            parts = line.strip().split(",")
            matches_dict[int(parts[0])]=int(parts[1])
    return matches_dict

def tranform_attributes(data_dicto):
    #831 attributes
    result = defaultdict(list)
    for k,v in data_dicto.items():
        value = [0]*831
        for a in v:
            value[a] = 1
        result[k]=value
    
    return result
            
    
        

        


data_dict = read_and_parse_data(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\user_features_0based.txt")

matches_dict = read_dict(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\data\matches_dict.txt") 
#matches_dict = get_dict_matches(interactions1235)




# Write dictionary to a text file
""" with open('warm/data/matches_dict.txt', 'w') as file:
    for key, value in matches_dict.items():
        file.write(f'{key},{value}\n')
 """




def get_graphs(matches_dict, data_dict):
    user_has_attribute = []
    attribute_has_matches = []
    attribute = 5
    for i in matches_dict.keys():
        if attribute in data_dict[i]:
            user_has_attribute.append(1)
        else:
            user_has_attribute.append(0)
        matches = matches_dict[i]
        attribute_has_matches.append(int(matches))

    
    """ user_has_attribute = user_has_attribute[:682]
    attribute_has_matches = attribute_has_matches[:682] """

    plt.scatter(user_has_attribute, attribute_has_matches, color='red')
    plt.title('Attribute '+str(attribute)+" from user with no match")
    plt.xlabel('Has attribute(boolean)')
    plt.ylabel('Interactions')

    plt.show()

    # Create a DataFrame from the lists
    violin_data = pd.DataFrame({'has_attribute': user_has_attribute, 
                        'has_matches': attribute_has_matches})
    


    plt.figure(figsize=(10, 6))
    sns.violinplot(x='has_attribute', y='has_matches', data=violin_data)


    plt.title('Violin Plot of attribute' + str(attribute)+" from a user with no match")
    plt.xlabel('Has attribute')
    plt.ylabel('Has matches')
    plt.show()

def get_int_att(matches_dict, data_dict):
    
    result = read_dict(r"C:\Users\User\OneDrive - UNIVERSIDAD DE SEVILLA\IRIT\eval\warm\data\int_attt.txt")
    plt.scatter(result.keys(), result.values(), color='green')
    plt.show()

def binary_classification():
    
    x, y= [],[]
    binary_att = tranform_attributes(data_dict)
    size1, size2 = 0,0
    count_elements = 0
    for user, inte in matches_dict.items():
        x.append(binary_att[user])
        
        if inte > 0:
            y.append(1)
        else:
            y.append(0)
        """ if count_elements >=1000000:
            break  """  
        count_elements+=1
        size1 = len(binary_att[user])
    
    clf = tree.DecisionTreeClassifier(max_depth=3)
     # Apply undersampling using RandomUnderSampler
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    x, y = oversampler.fit_resample(x, y)


    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
        test_size=0.3, random_state=42)
    clf = clf.fit(x_train, y_train)

        # Predict
    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))

    
    """ plt.figure(figsize=(14, 9))
    tree.plot_tree(clf,max_depth=3)
    ax = plt.gca() """

    importances = clf.feature_importances_

    for i in importances:
        if i>0.1:
            print("Attribute: ", str(np.where(importances == i)[0][0]), 
                  "importance: ", str(i))

    """ # Modify leaf node size
    for node in ax.get_children():
        if isinstance(node, plt.Annotation):
            # Adjust the size of the leaf nodes (circles)
            node.set_fontsize(13)  # Adjust the fontsize as needed

    #plt.show() """

    #plot_decision(x,y,clf)

    dot_data = tree.export_graphviz(clf, out_file=None,  
                     filled=True, rounded=True,  
                     special_characters=True)  
    graph = str(graphviz.Source(dot_data))
    with open("Tree.txt", "w") as file:
    # Write the string variable to the file
        file.write(graph)
    
    
    return [x,y]
    # console -> [[0. 1.]] which means 0 probability for 0, 1 for 1
def k_classification():
    """ X = [[0, 0], [1, 1]]
    Y = [0, 1] """
    x, y= [],[]
    binary_att = tranform_attributes(data_dict)
    size1, size2 = 0,0
    count_elements = 0
    for user, inte in matches_dict.items():
        x.append(binary_att[user])
        y.append(inte)
        
        """  if count_elements >=19999:
            break """
        count_elements+=1
    
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(x, y)
    
    plt.figure(figsize=(14, 9))
    tree.plot_tree(clf,max_depth=3)
    ax = plt.gca()

# Modify leaf node size
    for node in ax.get_children():
        if isinstance(node, plt.Annotation):
            # Adjust the size of the leaf nodes (circles)
            node.set_fontsize(13)  # Adjust the fontsize as needed

    #plt.show()

    #plot_decision(x,y,clf)

    importances = clf.feature_importances_

    for i in importances:
        if i>0.1:
            print("Attribute: ", str(np.where(importances == i)[0][0]), 
                  "importance: ", str(i))

    dot_data = tree.export_graphviz(clf, out_file=None,  
                     filled=True, rounded=True,  
                     special_characters=True)  
    graph = str(graphviz.Source(dot_data))
    with open("k_tree.txt", "w") as file:
    # Write the string variable to the file
        file.write(graph)
        
        # Get the class labels
    class_labels = clf.classes_
    print(class_labels)

def random_forest():
    x, y= [],[]
    rf_classifier = RandomForestClassifier(random_state=42, max_depth=2)
    rf_classifier5 = RandomForestClassifier(random_state=42, max_depth=5)


    binary_att = tranform_attributes(data_dict)
    
    
    for user, inte in matches_dict.items():
        x.append(binary_att[user])
        y.append(inte)

    
     
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
        test_size=0.3, random_state=42)

    
    rf_classifier.fit(x_train, y_train)
    rf_classifier5.fit(x_train, y_train)

        
    # Make predictions on train and test sets
    
    y_pred = rf_classifier.predict(x_test)
    y_pred5 = rf_classifier5.predict(x_test)

    # Calculate and print accuracy for the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on Test Set by RF depth 2:", test_accuracy)

    test_accuracy = accuracy_score(y_test, y_pred5)
    print("Accuracy on Test Set by RF depth 5:", test_accuracy)
    


    x_plot = list(range(1,164675))
    print(f"Length of x_plot: {len(x_plot)}")
    # Plot the results
    plt.figure()
    plt.scatter(x_plot[:1500], y_test[:1500], s=16, edgecolor="black", c="darkorange", label="data")
    plt.plot(x_plot[:1500], y_pred[:1500], color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(x_plot[:1500], y_pred5[:1500], color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("nº interactions")
    plt.title("Random Forest")
    plt.legend()
    plt.show()

def accuracy_binary_classification():
    x, y= [],[]
    binary_att = tranform_attributes(data_dict)
    size1, size2 = 0,0
    count_elements = 0
    for user, inte in matches_dict.items():
        x.append(binary_att[user])
        
        if inte > 0:
            y.append(1)
        else:
            y.append(0)
        """ if count_elements >=1000000:
            break  """  
        count_elements+=1
        size1 = len(binary_att[user])
    
    clf = tree.DecisionTreeClassifier(max_depth=3)
    x_initial, y_initial = x,y
    # Apply undersampling using RandomUnderSampler
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    x, y = oversampler.fit_resample(x, y)

    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_initial, y_initial, 
        test_size=0.3, random_state=42)
    
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, 
        test_size=0.3, random_state=42)
    
    clf = clf.fit(x_train2, y_train2)

        # Predict
    y_pred = clf.predict(x_test)

    get_accuracy(clf, x_test, y_test, y_pred)

    
    


def get_accuracy(clf,x_test, y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print("\n"+"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    disp = ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test, 
                                                display_labels=["Class 0", "Class 1"])
        
        # Customize the plot (optional)
    disp.ax_.set_title("Confusion Matrix")

    # Display the plot
    plt.show()

def get_cross_validation(clf, x, y):
    scores = cross_val_score(clf, x, y, cv=3)
    print("Cross-validation scores:", scores)
    print("Average accuracy:", scores.mean())



""" 
def plot_decision(x,y):
        # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)

        # Train the classifier
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf.fit(X_pca, y)

    # Parameters
    n_classes = 18
    plot_colors = plt.cm.get_cmap('tab20', n_classes)
    plot_step = 0.02

    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_pca,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel='PCA Component 1',
        ylabel='PCA Component 2',
    )

    # Plot the training points
    for i in range(n_classes):
        idx = np.where(y == i)
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            c=np.array([plot_colors(i)]),
            label=f'Class {i}',
            edgecolor="black",
            s=20,
        )

    plt.title("Decision surface of decision trees trained on PCA-reduced features")
    plt.legend(loc="lower right", borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()
 """

#get_graphs(matches_dict, data_dict)
#get_int_att(matches_dict, data_dict)
#binary_classification()
#k_classification()
#random_forest()

accuracy_binary_classification()
