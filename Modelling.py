from pylab import plot,show
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from sklearn import decomposition
from sklearn.grid_search import GridSearchCV



def my_tokenizer(s):
    return s.split()
def makeX(Text,X_min_df=.0001):
    X_vectorizer = TfidfVectorizer(min_df = X_min_df, sublinear_tf=True, max_df=0.9,stop_words='english')
    X = X_vectorizer.fit_transform(Text)
    return coo_matrix(X), X_vectorizer

def makeY(Tags, Y_min_df=.0001):
    Y_vectorizer = CountVectorizer(tokenizer = my_tokenizer, min_df = Y_min_df, binary = True)
    Y = Y_vectorizer.fit_transform(Tags)
    return Y, Y_vectorizer
def df_to_preds(dfmatrix, k = 5):
    predsmatrix = np.zeros(dfmatrix.shape)
    for i in range(0, dfmatrix.shape[0]):
        dfs = list(dfmatrix[i])
        if (np.sum([int(x > 0.0) for x in dfs]) <= k):
            predsmatrix[i,:] = [int(x > 0.0) for x in dfs]
        else:
            maxkeys = [x[0] for x in sorted(enumerate(dfs),key=operator.itemgetter(1),reverse=True)[0:k]]
            listofzeros = [0] * len(dfs)
            for j in range(0, len(dfs)):
                if (j in maxkeys):
                    listofzeros[j] = 1
            predsmatrix[i,:] = listofzeros
    return predsmatrix
def probs_to_preds(probsmatrix, k = 5):
    predsmatrix = np.zeros(probsmatrix.shape)
    for i in range(0, probsmatrix.shape[0]):
        probas = list(probsmatrix[i])
        if (np.sum([int(x > 0.01) for x in probas]) <= k):
            predsmatrix[i,:] = [int(x > 0.01) for x in probas]
        else:
            maxkeys = [x[0] for x in sorted(enumerate(probas),key=operator.itemgetter(1),reverse=True)[0:k]]
            listofzeros = [0] * len(probas)
            for j in range(0, len(probas)):
                if (j in maxkeys):
                    listofzeros[j] = 1
            predsmatrix[i,:] = listofzeros
    return predsmatrix
def opt_params(clf_current, params):
    model_to_set = OneVsRestClassifier(clf_current)
    grid_search = GridSearchCV(model_to_set, param_grid=params)

    print("Performing grid search on " + str(clf_current))
    print("parameters:")
    print(params)
    grid_search.fit(X_train, Y_train.toarray())
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    gs = grid_search.grid_scores_
    ret = [(i[0], i[1]) for i in gs]
    return best_parameters, ret
def benchmark(clf_current):
    print('_' * 80)
    print("Test performance for: ")
    clf_descr = str(clf_current).split('(')[0]
    print(clf_descr)
    t0 = time()
    classif = OneVsRestClassifier(clf_current)
    classif.fit(X_train, Y_train.toarray())
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time()
    if hasattr(clf_current,"decision_function"):
        dfmatrix = classif.decision_function(X_test)
        score = metrics.f1_score(Y_test.toarray(), df_to_preds(dfmatrix, k = 5))
    else:
        probsmatrix = classif.predict_proba(X_test)
        score = metrics.f1_score(Y_test.toarray(), probs_to_preds(probsmatrix, k = 5))
        
    test_time = time() - t0

    
    print("f1-score:   %0.7f" % score)
    print("test time:  %0.3fs" % test_time)

    print('_' * 80)
    return clf_descr, score, train_time, test_time
import warnings
warnings.filterwarnings("ignore")

Y, vectorizer2 = makeY(Tags, Y_min_df=int(10))
X, vectorizer1 = makeX(Text, X_min_df=int(10))
X_current = X
X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)

results = []
classlist = [
(Perceptron(), {'estimator__penalty': ['l1', 'elasticnet'],"estimator__alpha":[.001,.0001],'estimator__n_iter':[50]}), 
(PassiveAggressiveClassifier(), {'estimator__C':[.01,.1,1.0],'estimator__n_iter':[50]}),
(LinearSVC(), {'estimator__penalty': ['l1','l2'], 'estimator__loss': ['l2'],'estimator__dual': [False], 'estimator__tol':[1e-2,1e-3]}),
(SGDClassifier(), {'estimator__penalty': ['l1', 'elasticnet'],"estimator__alpha":[.0001,.001],'estimator__n_iter':[50]}),
(MultinomialNB(), {"estimator__alpha":[.01,.1],"estimator__fit_prior":[True, False]}),
(BernoulliNB(), {"estimator__alpha":[.01,.1],"estimator__fit_prior":[True, False]})
            ]

for classifier, params_to_optimize in classlist:
    best_params, gs = opt_params(classifier, params_to_optimize)
    results.append(benchmark(best_params['estimator']))
def plot_results(current_results, title = "Score"):
    indices = np.arange(len(current_results))
    
    results2 = [[x[i] for x in current_results] for i in range(4)]
    
    clf_names, score, training_time, test_time = results2
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(training_time)
    
    plt.figure(1, figsize=(14,5))
    plt.title(title)
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.xlabel('Mean F1 Score (time values are indexed so max training time = 1.0)')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    
    plt.show()
def topic_transform(X, n_topics = 10, method = "SVD"):
    
    topics = decomposition.TruncatedSVD(n_components=n_topics).fit(X)
    X_topics = topics.transform(X)
    for i in range(0,X_topics.shape[0]):
        theline = list(X_topics[i])
        # following line is only important for SVD
        theline = [(x * int(x > 0)) for x in theline]
        topic_sum = np.sum(theline)
        X_topics[i] = list(np.divide(theline,topic_sum))
    return X_topics, topics
def print_topics(topics,vectorizer1,n_top_words = 12):
    # Inverse the vectorizer vocabulary to be able
    feature_names = vectorizer1.get_feature_names()
    
    for topic_idx, topic in enumerate(topics.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

X_svd, svd = topic_transform(X, n_topics = 20,method = "SVD")
print_topics(svd,vectorizer1)

X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)

results_svd = []

X_current = X_svd
X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)

for classifier, params_to_optimize in classlist:
   best_params, gs = opt_params(classifier, params_to_optimize)
   results_svd.append(benchmark(best_params['estimator']))

plot_results(results_svd, title = "Classifier F1 Results on 20-topic SVD model output")
svd_results = []

classif_tuple = SGDClassifier(), {'estimator__penalty': ['l1'],"estimator__alpha":[.0001,.001],'estimator__n_iter':[50]}

num_topics_list = [10,20,30,40,50,60,70,80,90,100]
for num_topics in num_topics_list:
    #print 'Now testing for NMF with ' + str(num_topics) + ' topics'
    #X_nmf, nmf = topic_transform(X, n_topics = num_topics, method = "NMF")
    #X_current = X_nmf
    #X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)
    #nmf_results.append(benchmark(SGDClassifier(penalty = 'l1',alpha = .0001, n_iter = 50))[1])
    
    print 'Now testing for SVD with ' + str(num_topics) + ' topics'
    X_svd, svd = topic_transform(X, n_topics = num_topics, method = "SVD")
    X_current = X_svd
    X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)
    svd_results.append(benchmark(SGDClassifier(penalty = 'l1',alpha = .0001, n_iter = 50))[1])
plt.plot(num_topics_list,svd_results,'b',label="svd")
plt.legend(loc='best')
plt.ylabel('mean f1 score')
plt.xlabel('number of topics')
plt.title('F1 score vs. number of topics')    
Num_Tags = [(x.count(' ') + 1) for x in Tags]
Lens = [(len(x) + 1) for x in Text]
logLens = np.log(Lens)

logLens = np.array(logLens).reshape(-1, 1)

regress = LinearRegression()
regress.fit(logLens, Num_Tags)
thepreds = list(regress.predict(logLens))

plt.figure(1)
plt.hist(thepreds)
plt.xlabel('predicted num_tags')
plt.ylabel('number of posts')
plt.title('fitted values for regression of num_tags against log(post length)')

print 'r-squared for regression against log(post length)'
print pearsonr(thepreds,Num_Tags)[0]**2

regress = LinearRegression()


plt.figure(2)
plt.hist(thepreds)
plt.xlabel('predicted num_tags')
plt.ylabel('number of posts')
plt.title('fitted values for regression of num_tags against nmf vector')

print 'r-squared for regression against nmf vector'
print pearsonr(thepreds,Num_Tags)[0]**2

regress = LinearRegression()
regress.fit(X, Num_Tags)
thepreds = list(regress.predict(X))

plt.figure(3)
plt.hist(thepreds)
plt.xlabel('predicted num_tags')
plt.ylabel('number of posts')
plt.title('fitted values for regression of num_tags against tf-idf vector')

print 'r-squared for regression against tf-idf vector'
print pearsonr(thepreds,Num_Tags)[0]**2
