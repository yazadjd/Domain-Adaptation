# Domain-Adaptation

### Overview

A central goal to machine learning is generalization: from a small observed sample of data from
a distribution (the “training set”) can we learn a good model which describes instances from
that distribution (as measured using a “test set”). Most often we assume that the training and
test distribution are the same, and that instances are drawn independently. That is, the data is
“i.i.d.”. This is the assumption behind the models and learning algorithms we have covered in
the subject.

In practise, however, data is often a lot messier: The intended test scenario is often different
to the training (or development) settings, that is, these datasets follow different distributions.
Consider, for example, a handwriting recognition system trained to recognise characters
written by a dozen different individuals, that is then deployed to recognise handwriting from
another unseen person. If this test user has a different writing style, uses a different pen, or
various other quirks that make their writing different from the training users, the predictive
performance of the system will degrade. In general, when labelled testing samples do not
match well with the sampling on which the learner has been trained, the system will not be
able to generalize well to these new samples. This raises the challenge of developing machine
learning methods which generalize well to both similar (“in-domain”) and dissimilar (“out-ofdomain”)
instances to those seen in training.

This problem goes by various names, such as domain mismatch or covariate shift. Transfer
learning or its specialization, domain adaptation, are methods for addressing the problem.

### Tasks

Task 1.1: Develop and evaluate baseline methods

In the FEDA paper, the authors consider six domain adaption baseline approaches: SRCONLY,
TGTONLY, ALL, WEIGHTED, PRED and LININT. The task is to implement and evaluate these
baselines on then Schools dataset and report their mean squared error (MSE).

Task 1.2: Implement and evaluate FEDA

Implement the FEDA feature augmentation method according to the description in the FEDA
paper, section 3.The implementation should be sufficiently modular to allow multiple different learning algorithms to be
used, i.e., the learning approaches from Task 1.1. The method is evaluated on the Schools dataset, using at least two different learning algorithms.
All hyperparameters are fit carefully, and sensitivity to the hyperparameter settings are reported in the Report file.
A secondary experiment is also performed to determine the effect of the amount of training data in the target domain on test error.

Task 2: Domain Adaptation Extension
The next task is to research other techniques for domain adaptation and related problems in
transfer learning. This part implements the CORAL method. For more information refer to the Report.

### Files

There are six python (.py files) that correspond to the source code
of the Tasks in the project. The code is in Python 3.6 programming language.
All the code has been run on the Google Colab platform with dependencies on 
the following standard and machine learning libraries:

- Pandas
- sklearn
- matplotlib
- numpy
- math
- xgboost
- operator
- scipy
- statistics

Below is a brief description of each python notebook:

Initial_Experiments.py: This notebook contains code that experiments with the 
	initial feature sets and different models (Linear Regression, MLP,
	Decision Trees, Random Forests, XGBoost, etc) using the male domain as the
	target domain. The idea behind these initial experiments is only to conclude on
	which two models to use for the rest of the project and how the feature set
	performs without one hot encoding. The error rates generated in this notebook
	is not used in final reporting.

 Task_1.1_with_OneHotEncoding TGT = Male_Domain.py:
	This notebook contains code relevant to Task 1.1 of the Project where the 
	domain adaptation baseline approaches are implemented and evaluated on the
	schools data set when the target domain is male school, after one hot encoding
	categorical features.

Task_1.1_with_OneHotEncoding TGT = Female_Domain.py:
	This notebook contains code relevant to Task 1.1 of the Project where the 
	domain adaptation baseline approaches are implemented and evaluated on the
	schools data set when the target domain is female school, after one hot encoding
	categorical features.

Task_1.1_with_OneHotEncoding TGT = Mixed_Domain.py:
	This notebook contains code relevant to Task 1.1 of the Project where the 
	domain adaptation baseline approaches are implemented and evaluated on the
	schools data set when the target domain is mixed school, after one hot encoding
	categorical features.

Task 1.2_FEDA_with_Secondary_Experiment.py:
	This notebook contains code relevant to Task 1.2 of the Project where the FEDA
	feature augmentation method is implemented and evaluated on the School's data
	set for each of the domains according to the ALL baseline data condition.
	Initial feature set is one hot encoded. Experiments are also done by changing 
	hyperparameters of the neural network in order to report its sensitivity.
	Finally, a secondary experiment is also performed to determine the effect of
	the amount of training data in the target domain on test error.

Task_2_CORAL.py:
	This notebook contains code relevant to Task 2 of the Project where we implement
	the CORAL approach and evaluate the same on the School's Datset.
	Initial feature set is one hot encoded.
