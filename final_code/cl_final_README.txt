Begin by importing the complete "sent_5410" folder with its contents
into the directory from which you run all of your Python code.

PATH FORMATTING
In all of our testing, we were able to run the programs from
this folder without any need to change the path of the variables,
however, our IDE runs from the folder of the program file. If
your IDE differs, we have included the line number for the two
lines which need to have the path changed should there be any
errors.


LINEARSVC, LOGISTICREGRESSION, BERNOULLINB, and MULTINOMIALNB MODELS
In order to run the LinearSVC, LogisticRegression, BernoulliNB, and
MultinomialNB classifiers, all you need to do is run the "final_code.py"
program from the folder. These models will all run with tfidf transformation,
as they all run really well on a common set of data, and all provide
better results with tfidf transformation.
Train Data Path: Line 16
Test Data Path: Line 95


SVM.SVC MODEL
If you want to use the svm.SVC model, you will have to run the
"final_code_svc.py" program from the same folder.
*Warning* This program takes over 40 minutes to compile.
Though you can make it faster by commenting out two sections:
Lines 42 - 53
Lines 84 - 98
This will remove the tfidf transformation step and return only one set
of statistics
Train Data Path: Line 11
Test Data Path: Line 59


GAUSSIAN NAIVE BAYES MODEL
If you want to use the Gaussian Naive Bayes model, you will have to run
the  "final_code_GaussianNB.py" program from the same folder.
*Warning* This program takes over 20 minutes to compile.
This program will also not use tfidf transformation as we did not have
time to research the proper sequence for applying both tfidf transformation
and numpy array smoothing. Since this model was far from the most
accurate, we decided to devote the extra time elsewhere.
Train Data Path: Line 12
Test Data Path: Line 51
