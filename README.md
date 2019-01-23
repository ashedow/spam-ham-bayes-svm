# Anti spam analis

This progect is anothe one solution for spam (Junk Mail) filtering problem using supervised learning approaches (Naive Bayes algorithm and Support Vector Machines.)
In my decision, I do not use the methods described in

## Theory

### What is spam and how detectet it

Many antispam system, like rspamd or spamassasin, using score rules for calculate "spam weight" for every mail. 
Simple silly example:
* Mail has no title -5.0
* Mail has right SPF sign +1.0
* Mail has only one picture in the body -5.0

So, if he has -5 score for mark mail as spam and -10 for reject mail, this mail will be marked as spam.

#### Network

* DKIM
* SPF
* DMARK

#### On body

* Spam worlds
* mail contains one picture 
* mail conmtains only one link
* Mail without subject

### Naive Bayes 
Naive Bayes algorithm is one of the most well-known supervised algorithms. As we explained before, every machine learning algorithm has two phases; training and testing. Because of the nature of the supervised problem, Naive Bayes algorithm uses the dataset which has labeled samples. 

Naive Bayes algorithm is based on the [Bayesian Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) basically, the Naive Bayes algorithm uses word frequency in the email text. Training dataset has words, count of this words and class information for every sample. Basic dataset example has given below. Every row represents a single mail information.

1. Initially calculates the probabilities of ham and spam classes.
2. Calculates the probabilities of ham and spam for each word.
3. After the training phase, we calculate the probability of ham and spam for every sample using the words in that sample for the test phase. For this calculation, the equation used is given below.
![](prob_spam_or_ham.png)
4. Finally, pHam and pSpam are compared and ranked. And test sample is assigned to that class.

### SVM

Trying different SVM give the folowwing results, and  GridSearch can give the better result

## Spam detecting

**ham** - not spam.
**spam** - really spam.

1. Loading and prepare spam/ham base
---
/data
|--/spam - spam examples
|--/ham - ham examples
|--/test_spam - spam for test models
|--/test_ham - ham for test models
---

2. Create different models.

from sklearn create and test models:
MultinomialNB()
GaussianNB()
BernoulliNB()
svm.LinearSVC()
GridSearchCV(svm.SVC()

3. Pickle the best model

TODO:
* Testing code
* Optimisation process (treads) and memory usage
* Add cheking all mail, with headers and so on
* Create implementation for rspamd