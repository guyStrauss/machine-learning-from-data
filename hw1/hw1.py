###### Your ID ######
# ID1: 207253899
# ID2: 211482559
#####################

# imports 
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    X_avg = np.mean(X, axis=0)
    X_var = np.std(X, axis=0)
    X_normalized = (X - X_avg) / X_var

    Y_avg = np.mean(y, axis=0)
    Y_var = np.std(y, axis=0)
    y_normalized = (y - Y_avg) / Y_var

    return X_normalized, y_normalized


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    return np.c_[np.ones(X.shape[0]), X]


def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """
    J = 1 / (2 * X.shape[0]) * np.sum((np.sum(theta * X, axis=1) - y) ** 2)
    return J


def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    for _ in range(num_iters):
        gardient = 1 / X.shape[0] * np.sum(((np.sum(theta * X, axis=1) - y) * X.T),
                                           axis=1)
        theta -= eta * gardient
        J_history.append(compute_loss(X, y, theta))

    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    tol = 1e-15
    s_inv = np.array([1.0 / si if si > tol else 0.0 for si in s])

    S_inv = np.diag(s_inv)

    X_pinv = np.dot(Vt.T, np.dot(S_inv, U.T))

    pinv_theta = np.dot(X_pinv, y)

    return pinv_theta


def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than epsilon. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    J_history = []  # Use a python list to save the loss value in every iteration
    theta = theta.copy()
    for _ in range(max_iter):
        gradient = 1 / X.shape[0] * np.sum(
            ((np.sum(theta * X, axis=1, dtype="float64") - y) * X.T),
            axis=1, dtype="float64")
        theta -= eta * gradient
        loss = compute_loss(X, y, theta)
        if len(J_history) > 0 and abs(loss - J_history[-1]) < epsilon:
            break
        J_history.append(loss)

    return theta, J_history


def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using 
    the training dataset. Maintain a python dictionary with eta as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """

    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2,
            3]
    eta_dict = {}  # {eta_value: validation_loss}
    for eta in etas:
        theta, _ = gradient_descent(X_train, y_train,
                                                   np.random.rand(X_train.shape[1]), eta,
                                                   iterations)
        loss = compute_loss(X_val, y_val, theta)
        eta_dict[eta] = loss
    return eta_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    if X_train.shape[1] < 5:
        return list(range(X_train.shape[1]))
    bias_train = np.ones((X_train.shape[0], 1))
    bias_val = np.ones((X_val.shape[0], 1))

    selected_features = []
    remaining_features = list(range(X_train.shape[1]))

    while len(selected_features) < 5:
        best_loss = np.inf
        best_feature = None

        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_curr = np.hstack((bias_train, X_train[:, current_features]))
            X_val_curr = np.hstack((bias_val, X_val[:, current_features]))

            theta_init = np.random.rand(X_train_curr.shape[1])
            theta, _ = gradient_descent_stop_condition(X_train_curr, y_train,
                                                       theta_init, best_eta,
                                                       iterations)
            loss = compute_loss(X_val_curr, y_val, theta)

            if loss < best_loss:
                best_loss = loss
                best_feature = feature

        if best_feature is None:
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    original_columns = df_poly.columns
    for column in original_columns:
        new_name = column + "^2"
        df_poly = df_poly.assign(**{new_name: df_poly[column] ** 2})
    for column in original_columns:
        for column2 in original_columns:
            if column == column2:
                continue
            first, second = sorted([column, column2])
            col_name = first + "*" + second
            if col_name in df_poly.columns:
                continue

            df_poly = df_poly.assign(**{col_name: df_poly[first] * df_poly[second]})
    assert len(df_poly.columns) == len(original_columns) * 2 + len(original_columns) * (
            len(original_columns) - 1) / 2
    return df_poly
