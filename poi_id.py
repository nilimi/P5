#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# started with all features but email_address

features_list = ['poi','to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi', 'salary', 
'deferral_payments', 'total_payments', 'loan_advances', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 
'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.

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

#print 'Feature list with created features:'
#print features_list

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from tester import test_classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

names = ["Decision Tree", "AdaBoost", "Naive Bayes"]

classifiers = [
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
]

for name, clf in zip(names, classifiers):
    #Instantiate the various objects we'll need
    selector = SelectKBest()
    # build the pipeline, chaining the steps together. 
    pipe = Pipeline([
        ("select", selector), #select the best k features (3 here)
        (name, clf)]) #finally, take the output of all of the following 
                        #transformations and train a classifier
    pipe.fit(features, labels) #fit the pipeline on your data.
    print name
    test_classifier(pipe, my_dataset, features_list, folds = 10)
    print '**********************************************************'
	
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
	
pipe = Pipeline([
        ("select", selector), 
        ("clf", DecisionTreeClassifier())])  


param_grid = {
   "select__k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
               18, 19, 20, 21],
    "clf__criterion": ['gini', 'entropy'],
    'clf__max_features': ['sqrt', 'log2'],
    'clf__min_samples_split': [2, 5, 10, 20, 100]
}
sss = StratifiedShuffleSplit()
search = GridSearchCV(pipe, param_grid, cv=sss, scoring="f1")
#first we pass the pipeline, then the parameter grid we specified above,
#then the cross validation strategy (similar to train_test_split). 
#Here, we use StatifiedShuffleSplit,
#because that's what the tester function uses. Finally, we pass how we
#want to score the different combinations. Since we care about f1 score for
#this project, we'll use that.
search.fit(features, labels)
print 'search.best_params_'
print search.best_params_ #the parameter combination that together got the best f1 score
print 'search.best_estimator_'
print search.best_estimator_ #a fitted pipeline with those best parameters


selector = SelectKBest(k=8)
selector.fit_transform(features, labels)
selector_features=[features_list[1:][i] for i in selector.get_support(indices=True)]
combined = zip(selector_features, selector.scores_)
combined.sort( reverse=True, key= lambda x: x[1])
print combined

clf = DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list, folds = 100)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)