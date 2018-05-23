# Detection of Cerebral Hemorrhages With Supervised Learning

## Introduction

Acute ischemic stroke is a condition where blood flow to the brain is impeded by clots, which usually consist of plaque deposits (American Heart Association). Since strokes are the fifth leading cause of death in the United States, cerebrovascular ischemia prevention is of great significance (Impact of Stroke). Reperfusion therapy remains the most effective treatment for ischemic stroke, but brings the risk of hemorrhagic transformation (HT), or bleeding of brain tissue. In _Prediction of Hemorrhagic Transformation Severity in Acute Stroke from Source Perfusion MRI_, Yu et al. compare the performance of several classification models in machine learning to predict HT so that medical professionals can better evaluate the risks associated with reperfusion therapy. 

The purpose of this project is to compare the accuracy and speed of various classifiers in detecting cerebral hemorrhages in MRI data. The data was generously provided by Professor Fabien Scalzo of the Computer Science and Neuroscience departments at UCLA. It was processed from images into csv format by Yu et al. 

### Data set
- Size: 50,000 instances from MRI images
- Format: .csv
- Features: 618 – see Data set Readme in *./data*
- Positive/Negative labels: 25018/24982
	- accuracy is a suitable metric since the data is balanced

## Binary Classifiers Tested
- Support Vector Machine (SVM)
- Ensemble Classifiers (based on Decision Trees)
	- Bagging
	- Adaboost
	- Gradient Tree Boosting
	- Random Forest
	- Extremely Randomized Trees
- Stochastic Gradient Descent
- Logistic Regression
- K-nearest neighbors
- Nearest Centroid
- Multi-layer Perceptron (MLP) Classifier [neural network]
	
## Methods

Scikit-learn was used to implement all of the classifiers above using the built-in models. Keras was used to create a second custom neural network for evaluation as well as a Logistic Regression classifier. 

To implement each classifier, the input data was first loaded into a Jupyter notebook and split up 70%/30% for the training/testing sets. To improve the performance of the classifiers, the data was then feature-scaled using Scikit-learn's StandardScaler. Afterward, the dimensionality of the data was reduced using a Truncated SVD algorithm to reduce the amount of data fed into each classifier and improve performance.

### Development Environment

Development was completed using Google Colaboratory, which enables editing of Jupyter notebooks in the cloud. This was chosen because of its greater memory (12GB) and also for its GPU accelerated runtime. Since Tensorflow was selected as the backend for Keras, the library could take advantage of this runtime to improve performance. 

## Results
Mouse over the bars that go over the charts to see their values

<iframe width="1401.9203163017032" height="864.5" seamless frameborder="0" scrolling="no" src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQwLw6ia58vlLh1i2i4zxtcn2HNFwhS29XNx-lHCjCoGjQwTGju-5-_yQ4t-P_Ksq9t2urqHQQvAOQr/pubchart?oid=2084364792&amp;format=interactive"></iframe>

---
## Technical References
- General Machine Learning
	- [Is there a rule-of-thumb for how to divide a dataset into training and validation sets? (StackOverflow)](https://stackoverflow.com/questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
	- RandomizedSearchCV
		- [Randomized Parameter Optimization](http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)
		- [scipy.stats.expon - generates distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html)
- SVM
	- [Parameter Estimation using grid search with cross-validation](http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html)
- Ensemble Methods
	- [What is the difference between gradient boosting and adaboost? (Quora)](https://www.quora.com/What-is-the-difference-between-gradient-boosting-and-adaboost)
	- [Understanding Boosted Trees Models](https://sadanand-singh.github.io/posts/boostedtrees/#adaboost-classifier-in-python)
	- [What is the difference between Bagging and Boosting?](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
	- Adaboost
		- [Tuning adaboost (StackExchange)](https://stats.stackexchange.com/questions/303998/tuning-adaboost?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
		- [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
- Linear Models
	- Stochastic Gradient Descent
		- [Tips on Practical Use](http://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use)
		- [Cross Validating SGD](https://gist.github.com/tobigue/3188762)
- Neighest Neigbors
	[scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html#choice-of-nearest-neighbors-algorithm)
- Neural Networks
	- [Epoch vs iteration when training neural networks (StackOverflow)](https://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks)
	- [What's the difference between convolutional and recurrent neural networks? (Stack Overflow)](https://stackoverflow.com/questions/20923574/whats-the-difference-between-convolutional-and-recurrent-neural-networks?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
	- [Tradeoff batch size vs. number of iterations to train a neural network (StackOverflow)](https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network)
	- Keras
		- [Getting started with the Keras Sequential model](https://keras.io/getting-started/sequential-model-guide/)
			- example code for a binary classification MLP network
		- [Core Layers](https://keras.io/layers/core/)

## Sources
[1] **Impact of Stroke** (Stroke statistics). (2016, June 6). Retrieved May 6, 2018, from http://www.strokeassociation.org/STROKEORG/AboutStroke/Impact-of-Stroke-Stroke-statistics_UCM_310728_Article.jsp

[2] Y. Yu, D. Guo, M. Lou, D. S. Liebeskind and F. Scalzo, **Prediction of Hemorrhagic Transformation Severity in Acute Stroke from Source Perfusion MRI**, in _IEEE Transactions on Biomedical Engineering_.

