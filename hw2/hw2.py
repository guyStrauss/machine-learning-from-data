import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    labels = data[:, -1]
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        labels = self.data[:, -1]
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        
        values, _ = np.unique(self.data[:, feature], return_counts=True)
        total_size = len(self.data)
        impurity_before = self.impurity_func(self.data)
        split_info = 0

        for v in values:
            subset = self.data[self.data[:, feature] == v]
            groups[v] = subset
            weight = len(subset) / total_size
            goodness += weight * self.impurity_func(subset)
            if self.gain_ratio:
                split_info -= weight * np.log2(weight)

        info_gain = impurity_before - goodness

        if self.gain_ratio:
            if split_info == 0:
                return 0, groups
            return info_gain / (split_info), groups
        
        return info_gain, groups
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        weight = len(self.data) / n_total_sample
        self.feature_importance = weight * self.goodness_of_split(self.feature)[0]
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        if self.depth >= self.max_depth or len(np.unique(self.data[:, -1])) == 1:
            self.terminal = True
            return

        best_feature = None
        best_goodness = 0
        best_groups = {}

        for feature in range(self.data.shape[1] - 1):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_feature = feature
                best_goodness = goodness
                best_groups = groups

        if best_goodness == 0 or best_feature is None:
            self.terminal = True
            return

        # Chi-square pruning
        if self.chi < 1:
            total = len(self.data)
            observed = []
            expected = []
            class_labels, _ = np.unique(self.data[:, -1], return_counts=True)
            overall_dist = {label: np.sum(self.data[:, -1] == label) / total for label in class_labels}

            for subset in best_groups.values():
                subset_size = len(subset)
                subset_dist = {label: np.sum(subset[:, -1] == label) for label in class_labels}
                for label in class_labels:
                    observed.append(subset_dist[label])
                    expected.append(overall_dist[label] * subset_size)

            chi_square = np.sum((np.array(observed) - np.array(expected)) ** 2 / (np.array(expected)))
            df = (len(best_groups) - 1) * (len(class_labels) - 1)
            if df == 0 or chi_square < chi_table.get(df, {}).get(self.chi, 100001):
                self.terminal = True
                return

        self.feature = best_feature
        for val, subset in best_groups.items():
            child = DecisionNode(
                data=subset,
                impurity_func=self.impurity_func,
                depth=self.depth + 1,
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            child.split()
            self.add_child(child, val)

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the training data used to construct the tree
        self.root = None # the root node of the tree
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio #
        
    def depth(self):
        return self.root.depth

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
        )
        self.root.split()

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        node = self.root
        while not node.terminal:
            feature_val = instance[node.feature]
            if feature_val in node.children_values:
                i = node.children_values.index(feature_val)
                node = node.children[i]
            else:
                break
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        correct = 0
        for row in dataset:
            if self.predict(row) == row[-1]:
                correct += 1
        return (correct / len(dataset)) * 100
        

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = DecisionTree(X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))

    return training, validation


def chi_pruning(X_train, X_validation):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    for chi in [0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(X_train, impurity_func=calc_entropy, chi=chi, gain_ratio=True)
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_validation))
        depth.append(tree.depth())
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    if node.terminal:
        return 1
    return 1 + sum(count_nodes(child) for child in node.children)
