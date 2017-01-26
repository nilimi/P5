#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive', 'restricted_stock',
                 'to_messages', 'from_poi_to_this_person','from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

#for k in data_dict.itervalues().next():
 #   if k <> 'poi' and k <> 'email_address':
    #      features_list.append(k)

#remove_features = ['loan_advances', 'director_fees', 
 #                  'restricted_stock_deferred', 'deferral_payment']
#try:
 #   for i in remove_features:
#      features_list.remove(i)
#except:
#    pass

import matplotlib.pyplot
import numpy
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
from_ratio = []
for name in data_dict:
    if data_dict[name]['from_messages']== 'NaN' \
    or data_dict[name]['from_messages'] == 0:
        from_ratio.append(0)
    else:
        from_ratio.append(float(data_dict[name]['from_this_person_to_poi'])/
                                float(data_dict[name]['from_messages']))
to_ratio = []
for name in data_dict:
    if data_dict[name]['to_messages']== 'NaN' \
    or data_dict[name]['to_messages'] == 0:
        to_ratio.append(0)
    else:
        to_ratio.append(float(data_dict[name]['from_poi_to_this_person'])/
                                float(data_dict[name]['to_messages']))
count = 0        
for name in data_dict:
    data_dict[name]['from_ratio'] = from_ratio[count]
    data_dict[name]['to_ratio'] = to_ratio[count]
    count += 1

my_dataset = data_dict
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )

features_list.append('from_ratio')
features_list.append('to_ratio')
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
pickle.dump(clf, open("my_classifier.pkl", "w") )

#!/usr/bin/pickle
##########tester

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

CLF_PICKLE_FILENAME = "my_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset.pkl"
FEATURE_LIST_FILENAME = "my_feature_list.pkl"

def dump_classifier_and_data(clf, dataset, feature_list):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME, "w") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
    test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()
	
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
pickle.dump(clf, open("my_classifier.pkl", "w") )

if __name__ == '__main__':
    main()
	
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
pickle.dump(clf, open("my_classifier.pkl", "w") )
	
if __name__ == '__main__':
    main()
	
	
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
	
from sklearn.feature_selection import SelectKBest
kb = SelectKBest(k=10)
kb.fit_transform(features,labels)
kb_features=[features_list[i+1] for i in kb.get_support(indices=True)]
#print kb_features

features_list = ['poi']
count = 0
for i in kb_features: 
    features_list.append(i)
    print i, kb.scores_[count]
    count += 1
#print features_list
    
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
	
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_features='auto')
pickle.dump(clf, open("my_classifier.pkl", "w") )	

if __name__ == '__main__':
    main()