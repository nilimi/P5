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



### Task 2: Remove outliers

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)



sys.path.append("../final_project/")
from tester import dump_classifier_and_data, test_classifier, \
load_classifier_and_data, main



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

my_dataset = data_dict
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
pickle.dump(clf, open("my_classifier.pkl", "w") )
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



clf = DecisionTreeClassifier()
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()

  
clf = GaussianNB()
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()

 

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
kb = SelectKBest(k=17)
kb.fit_transform(features,labels)
kb_features=[features_list[i+1] for i in kb.get_support(indices=True)]

features_list = ['poi']
count = 0
for i in kb_features: 
    if kb.pvalues_[count] <0.05:
        features_list.append(i)
        print count, i, kb.pvalues_[count]
        count += 1
#print features_list
    
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


clf = KNeighborsClassifier()
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()

    
clf = DecisionTreeClassifier()
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()
    
clf = DecisionTreeClassifier(min_samples_split=100)
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()
    
clf = DecisionTreeClassifier(min_samples_split=10)
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()

clf = DecisionTreeClassifier(criterion='entropy')
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()
    
clf = DecisionTreeClassifier(max_features='auto')
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()
    
    
clf = GaussianNB()
pickle.dump(clf, open("my_classifier.pkl", "w") )
if __name__ == '__main__':
    main()
    
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)