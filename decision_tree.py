from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_data():
    """Loads the Iris dataset.  

    Tries first with sklearn's built-in `load_iris`, otherwise downloads
    the dataset from seaborn's GitHub repo.  

    Returns:
        X (ndarray): feature matrix.  
        y (ndarray): label vector.  
        class_names (list): names of the classes.  
        feature_names (list): names of the features.  
    """
    try:
        data = load_iris()
        X = data.data
        y = data.target
        class_names = data.target_names
        feature_names = data.feature_names
    except:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
        df["species"] = df["species"].astype("category").cat.codes
        X = df.drop("species", axis=1).values
        y = df["species"].values
        class_names = df["species"].astype("category").cat.categories
        feature_names = df.columns[:-1].tolist()
    return X, y, class_names, feature_names


def plot_data(y, class_names):
    """Plots the class distribution as a pie chart.  

    Args:  
        y (ndarray): label vector.    
        class_names (list): names of the classes.  
    """
    class_counts = np.bincount(y)
    plt.figure(figsize=(6, 6))
    plt.pie(
        class_counts,
        labels=class_names,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Class Distribution")


def one_hot_encode(y):
    """Encodes labels into one-hot representation.  

    Args:
        y (ndarray): vector of integer labels.  

    Returns:
        Y (ndarray): one-hot encoded label matrix.  
    """
    Y = np.zeros((y.size, y.max() + 1))
    Y[np.arange(y.size), y] = 1
    return Y


def standardize(X):
    """Standardizes features by removing the mean and scaling to unit variance.  

    Args:
        X (ndarray): feature matrix.  

    Returns:
        X_std (ndarray): standardized feature matrix.  
    """
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sigma


def train_test_split(X, y, test_size=0.2, random_state=42):
    """Splits dataset into train and test sets using random permutation.  

    Args:
        X (ndarray): feature matrix.  
        y (ndarray): label vector.  
        test_size (float): fraction of samples used for testing.  
        random_state (int): random seed for reproducibility.  

    Returns:
        X_train, X_test, y_train, y_test (tuple of ndarrays).  
    """
    np.random.seed(random_state)
    indexes = np.random.permutation(len(X))
    train_size = int((1 - test_size) * len(X))
    train_idx, test_idx = indexes[:train_size], indexes[train_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


class Node:
    """A node in the decision tree.  ay

    Attributes:
        feature_index (int): index of the feature to split on.  
        threshold (float): threshold value for the split.  
        left (Node): left child node.  
        right (Node): right child node.  
        gain (float): gain fo the split.
        value (int): class label for leaf nodes.  
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTree:
    """A Decision Tree classifier.  

    Args:
        min_samples_split (int): minimum number of samples required to split a node.  
        max_depth (int): maximum depth of the tree.  
    """

    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        """Fits the decision tree to the training data.  

        Args:
            X (ndarray): feature matrix.  
            y (ndarray): label vector.  
        """
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        if (num_samples >= self.min_samples_split) and (depth < self.max_depth) and (len(unique_classes) > 1):
            best_gain = -1
            best_feature_index = None
            best_threshold = None
            best_splits = None

            for feature_index in range(num_features):
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    left_indices = X[:, feature_index] <= threshold
                    right_indices = X[:, feature_index] > threshold

                    if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                        continue

                    gain = self._information_gain(y, y[left_indices], y[right_indices])

                    if gain > best_gain:
                        best_gain = gain
                        best_feature_index = feature_index
                        best_threshold = threshold
                        best_splits = (left_indices, right_indices)

            if best_gain > 0:
                left_node = self._grow_tree(X[best_splits[0]], y[best_splits[0]], depth + 1)
                right_node = self._grow_tree(X[best_splits[1]], y[best_splits[1]], depth + 1)
                return Node(feature_index=best_feature_index, threshold=best_threshold, left=left_node, right=right_node, gain=best_gain)

        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = self._entropy(parent) - (weight_left * self._entropy(left_child) + weight_right * self._entropy(right_child))
        return gain
    
    def _entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return entropy
    
    def _most_common_label(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        most_common = class_labels[np.argmax(counts)]
        return most_common
    
    def predict(self, X):
        """Predicts class labels for the input samples.  

        Args:
            X (ndarray): feature matrix.  

        Returns:
            y_pred (ndarray): predicted class labels.  
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
    def evaluate(self, X, y, class_names):
        """Evaluates the model on the test data and prints metrics.  

        Args:
            X (ndarray): feature matrix.  
            y (ndarray): true label vector.  
            class_names (list): names of the classes.  
        """
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=class_names))        

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        accuracy = np.sum(y == y_pred) / len(y)
        print(f"Accuracy: {accuracy * 100:.2f}% \n")

if __name__ == "__main__":
    # Load data
    X, y, class_names, feature_names = load_data()

    # Plot data distribution
    plot_data(y, class_names)

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=5)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=5)

    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")

    # Initialize and train the decision tree
    dt = DecisionTree(min_samples_split=3, max_depth=5)
    dt.fit(X_train, y_train)

    # Evaluate on validation set
    print("Validation Set Evaluation:")
    dt.evaluate(X_val, y_val, class_names)

    # Evaluate on test set
    print("Test Set Evaluation:")
    dt.evaluate(X_test, y_test, class_names)

    plt.show()
        