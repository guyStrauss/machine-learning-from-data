import numpy as np

def add_bias_term(X):
    """
    Add a bias term to each sample of the input data.
    """

    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

class LogisticRegressionGD():
    """
    Logistic Regression Classifier.

    Fields:
    -------
    w_ : array-like, shape = [n_features]
      Weights vector, where n_features is the number of features.
    eta : float
      Learning rate (between 0.0 and 1.0)
    max_iter : int
      Maximum number of iterations for gradient descent
    eps : float
      minimum change in the BCE loss to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """
    
    def __init__(self, learning_rate=0.0001, max_iter=10000, eps=0.000001, random_state=1):
       
        # Initialize the weights vector with small random values
        self.random_state = random_state
        self.w_ = np.nan
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.class_names = None


    def predict_proba(self, X):
        """
        Return the predicted probabilities of the instances for the positive class (class 1)

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.

        Returns
        -------
        y_pred_prob : array-like, shape = [n_examples]
          Predicted probabilities (for class 1) for all the instances
        """
        class_1_prob = np.nan * np.ones(X.shape[0])

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################
        z = X @ self.w_
        class_1_prob = 1 / (1 + np.exp(-z))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return class_1_prob
        

    def predict(self, X, threshold=0.5):
        """
        Return the predicted class label according to the threshold

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.
        threshold : float, optional
          Threshold for the predicted class label.
          Predict class 1 if the probability is greater than or equal to the threshold and 0 otherwise.
          Default is 0.5. 
        """
        y_pred = np.nan * np.ones(X.shape[0])
    
        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################
        prob = self.predict_proba(X)
        y_pred = (prob >= threshold).astype(int)
        y_pred = np.where(y_pred == 0, self.class_names[0], self.class_names[1])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return y_pred

    
    def BCE_loss(self, X, y):
        """
        Calculate the BCE loss (not needed for training)

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Instance vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Class labels. 

        Returns
        -------
        BCE_loss : float
          The BCE loss.
          Make sure to normalize the BCE loss by the number of samples.
        """

        y_01 = np.where(y == self.class_names[0], 0, 1) # represents the class 0/1 labels
        loss = None
        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################
        prob = self.predict_proba(X)
        prob = np.clip(prob, 1e-15, 1 - 1e-15)  # To avoid log(0)
        loss = -np.mean(y_01 * np.log(prob) + (1 - y_01) * np.log(1 - prob))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return loss


    def fit(self, X, y):
        """ 
        Fit training data by minimizing the BCE loss using gradient descent.
        Updates the weight vector (field of the object) in each iteration using gradient descent.
        The gradient should correspond to the BCE loss normalized by the number of samples.
        Stop the function when the difference between the previous BCE loss and the current is less than eps
        or when you reach max_iter.
        Collect the BCE loss in each iteration in the loss variable.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Class labels.

        """

        # Initial weights are set in constructor
        # Initialize the cost history
        loss = []

        # make sure to use 0/1 labels:
        self.class_names = np.unique(y)
        y_01 = np.where(y == self.class_names[0], 0, 1)
        np.random.seed(self.random_state)
        self.w_ = 1e-6 * np.random.randn(X.shape[1])

        ###########################################################################
        # TODO: Implement the function in section below.                          #
        ###########################################################################
        for i in range(self.max_iter):
            prob = self.predict_proba(X)
            grad = X.T @ (prob - y_01) / X.shape[0]
            self.w_ -= self.learning_rate * grad

            current_loss = self.BCE_loss(X, y)
            loss.append(current_loss)

            if i > 0 and abs(loss[-2] - current_loss) < self.eps:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        

def select_learning_rate(X_train, y_train, learning_rates, max_iter):
    """
    Select the learning rate attaining the minimal BCE after max_iter GD iterations

    Parameters
    ----------
    X_train : {array-like}, shape = [n_samples, n_features]
      Training vectors, where n_samples is the number of samples and
      n_features is the number of features.
    y_train : array-like, shape = [n_samples]
      Class labels.
    learning_rates : list
      The list of learning rates to test.
    max_iter : int
      The maximum number of iterations for the gradient descent.

    Returns
    -------
    selected_learning_rate : float
      The learning rate attaining the minimal BCE after max_iter GD iterations.
    """
    # Initialize variables to keep track of the minimum BCE and the corresponding learning rate
    min_bce = float('inf')
    selected_learning_rate = None
    
    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    for lr in learning_rates:
        print(lr)
        model = LogisticRegressionGD(learning_rate=lr, max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)
        bce = model.BCE_loss(X_train, y_train)

        if bce < min_bce:
            min_bce = bce
            selected_learning_rate = lr
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_learning_rate


def cv_accuracy_and_bce_error(X, y, n_folds):
    """
    Calculate the accuracy and BCE error of the model using cross-validation.

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]
      Training samples, where n_samples is the number of samples and
      n_features is the number of features.
    y : array-like, shape = [n_samples]
      Target values.
    n_folds : int
      The number of folds for cross-validation.
    Returns 
    -------
    The function returns two lists: accuracies and BCE_losses.
    Each list contains the results for each of the n_folds of the cross-validation.
    """

    # Split the data into n_folds and initialize the lists for accuracies and BCE losses
    X_splits = np.array_split(X, n_folds)
    y_splits = np.array_split(y, n_folds)
    accuracies = []
    BCE_losses = []

    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    for i in range(n_folds):
        # Prepare validation fold
        X_val = X_splits[i]
        y_val = y_splits[i]

        # Prepare training folds (concatenate all folds except the i-th)
        X_train = np.concatenate([X_splits[j] for j in range(n_folds) if j != i], axis=0)
        y_train = np.concatenate([y_splits[j] for j in range(n_folds) if j != i], axis=0)

        # Train the model
        model = LogisticRegressionGD(max_iter=1000, learning_rate=0.01, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)
        acc = np.mean(y_pred == y_val)
        bce = model.BCE_loss(X_val, y_val)

        accuracies.append(acc)
        BCE_losses.append(bce)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return accuracies, BCE_losses


def calc_and_print_metrics(y_true, y_pred, positive_class):
    """
    Calculate and print the metrics for the LogisticRegression classifier.
    """
    # Calculate the metrics
    
    tp, fp, tn, fn = None, None, None, None
    tpr, fpr, tnr, fnr = None, None, None, None
    accuracy, precision, recall = None, None, None
    risk = None
    f1 = None

    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    y_true_bin = (y_true == positive_class).astype(int)
    y_pred_bin = (y_pred == positive_class).astype(int)

    # Confusion matrix components
    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    # Avoid division by zero
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    risk = 1 - accuracy
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # Print the metrics    
    print(f"#TP: {tp}, #FP: {fp}, #TN: {tn}, #FN: {fn}")
    print(f"#TPR: {tpr}, #FPR: {fpr}, #TNR: {tnr}, #FNR: {fnr}")
    print(f"Accuracy: {accuracy}, Risk: {risk}, Precision: {precision}, Recall: {recall}")
    print(f"F1: {f1}")



def fpr_tpr_per_threshold(y_true, positive_class_probs, positive_class="9"):
    """
    Calculate FPR and TPR of a given classifier for different thresholds

    Parameters
    ----------
    y_true : array-like, shape = [n]
      True class labels for the n samples
    positive_class_probs : array-like, shape = [n]
      Predicted probabilities for the positive class for the n samples
    positive_class : str, optional
      The label of the class to be considered as the positive class
    """
    fpr = []
    tpr = []
    # consider thresholds from 0 to 1 with step 0.01
    prob_thresholds = np.arange(0, 1, 0.01)
    y_true_binary = np.where(y_true == positive_class, 1, 0)
    
    ###########################################################################
    # TODO: Implement the function in section below.                          #
    ###########################################################################
    for threshold in prob_thresholds:
        # Generate binary predictions
        y_pred_binary = (positive_class_probs >= threshold).astype(int)

        # Confusion matrix components
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        # Compute TPR and FPR
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr.append(tpr_val)
        fpr.append(fpr_val)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fpr, tpr



