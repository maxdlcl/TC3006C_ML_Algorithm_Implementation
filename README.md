# Implementation of the Decision Tree algorithm

---

This repository contains an implementation of a **Decision Tree Classifier** built only with **NumPy** and **Pandas** (no scikit-learn, no advanced ML frameworks).

This implementation was tested using the [Iris Dataset](https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html), which contains of the three classes: Setosa, Versicolor, Virginica with the features for its Sepal length, Sepal width, Petal length, Petal width; and a total of 150 samples (50 per class).

After running the binay three plots are shown:

1. A pie chart showing the data distribution of the dataset.
2. A confusion matrix with the model's performance with the validation set.
3. A confusion matrix with the model's performance with the test set.

## Running the implementation

In order to execute the Decision Tree implementation, it is necessary to install the libraries used in the code.

Execute the following commands:
```
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install pandas
pip install numpy
```

Then, execute the file for the implementation with the following commands:

```
gith clone https://github.com/maxdlcl/TC3006C_ML_Algorithm_Implementation
cd TC3006C_ML_Algorithm_Implementation
python3 decision_tree.py
```

## Submission info

* Maximiliano De La Cruz Lima
* A01798048
* Submission date: September 5th, 2025
