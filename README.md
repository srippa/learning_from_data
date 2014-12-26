learning_from_data
==================

Selected topics in machine learning


During past years I got interested in learning various data science topics and I decided to summarize the lessons 
I learned as serie of [IPYthon notebooks](http://ipython.org/notebook.html) with added code sniplets and plots to 
demonstrate and highlight selected parts. In most cases I use the standard Python libraries for data 
science computations: [numpy](http://www.numpy.org/),[matplotlib](http://matplotlib.org/), 
[statsmodels](http://statsmodels.sourceforge.net/), [pandas](http://pandas.pydata.org/) and 
[scikit-learn](http://scikit-learn.org/stable/). 
Sometimes I will add new interesting libraries like, e.g. [seaborn](http://stanford.edu/~mwaskom/software/seaborn/). 

In few occasions I will implement a method from scratch. In such cases, I always prefer clarity of presentation over 
performance ornumerical stability. For example I invert the normal equations whensolving least-square problems which is a 
seriously flawed approach from numerical stability point of view.  Thus my homemade implementations are **not** 
recommended for practicalk usage - use code from a respected library like one of 
those mensionned above. 

## Index

* **C01_learning_from_data_introduction**: <font colour="green">Problem formulation, No-free-lunch theorems, Bias-Variance, PAC-learning, Overfitting, 
Curse of dimensionality, Model selection Cost functions</font>


## In preparation
* **Measuring performance** - Test error , Bias-Variance
* **Dimensionality reduction** Feature engineering, feature selection, PCA
* **Linear Regression** - Ordinary least sequares, ridge regression, LASSO, feature selection, PCA, radial basis functions
* **Linear classification** - Logistic regression, Optimization using Newton iterations, Gradient descent, perceptron , Stochastic Gradient Descent
* **SVM and RBF** - Linear separable case, Non linear separable case, non-separable case
* **Naive Bayes **
* **Decision trees** -- impurity measures, tree growing, prunning, bagging, random forests, boosting, feature selection with trees
* **Resources**) - Tools, Books, Courses, Software, Links
