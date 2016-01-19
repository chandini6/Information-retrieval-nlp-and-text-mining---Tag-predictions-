# Informationretrieval, nlp and textmining techiniques :  Multilabel classifications model


Tags predictions for stackoverflow data set


This data set is obtained from kaggle previous competitions which contains the posts, tags assigned to the questions and the title of the post. The data is cleaned by removing the code snippets, punctuation and numbers. Fequency disturbution of the words and tops tags are plotted. The post  are converted into term vectorixer and the tags are treated as one vs rest classifier. Naive bayes SGD , Linear svc are used for the prediction. The perfomance metric is choosen as F-1 score for different algorithms on testing and validation set
