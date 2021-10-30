import Eval_Matrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#import time



# a dummy function that just returns its input
def identity(x):
    return x

# decide on TF-IDF vectorization for feature
# based on the value of tfidf (True/False)
def tf_idf_func(tfidf):
    # TODO - change the values
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity, tokenizer = identity)

    else:
        vec = CountVectorizer(preprocessor = identity, tokenizer = identity)

    return vec

def SVM_Normal(trainDoc, trainClass, testDoc, testClass, tfIdf):

    # decides on TfidfVectorizer(True) or CountVectorizer(False)

    vec = tf_idf_func(tfIdf)

    #classifier = Pipeline( [('vec', vec),
     #                       ('cls', svm.SVC(kernel='linear', C=1.0))] )

    #classifier = Pipeline( [('vec', vec),
     #                       ('cls', DecisionTreeClassifier(max_depth=15))] )
    #classifier = Pipeline( [('vec', vec),
     #                       ('cls', MultinomialNB())] )
    classifier = Pipeline([('vec', vec),
                           ('cls', KNeighborsClassifier(n_neighbors=15))])

    # Here trainDoc are the documents from training set and trainClass is the class labels for those documents
    classifier.fit(trainDoc, trainClass)

    # Use the classifier to predict the class for all the documents in the test set testDoc
    # Save those output class labels in testGuess
    testGuess = classifier.predict(testDoc)


    # Just to know which version of Tfidf is being used
    tfIDF_type = "TfidfVectorizer" if(tfIdf) else "CountVectorizer"

    print("\n########### Default SVM Classifier For (", tfIDF_type, ") ###########")
    print('Training accuracy: ', classifier.score(trainDoc, trainClass))
    print('Testing accuracy: ', classifier.score(testDoc, testClass))

    title = "Linear SVM (C = 1.0)"
    #for evaluation
    Eval_Matrics.calculate_measures(classifier, testClass, testGuess, title)




