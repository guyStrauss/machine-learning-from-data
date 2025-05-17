import numpy as np


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    factorial_vectorized = np.vectorize(np.math.factorial)
    const = factorial_vectorized(k)
    return np.log(np.power(rate, k) * np.exp(-rate) / const)



def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    return np.sum(samples) / samples.shape[0]


def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    return lambda_mle + norm.ppf(alpha / 2) * np.sqrt(lambda_mle / n), lambda_mle + norm.ppf(1- alpha / 2) * np.sqrt(
        lambda_mle / n)


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    samples = np.asarray(samples)
    rates = np.asarray(rates)
    term1 = np.sum(samples) * (np.log(rates))
    term2 = - samples.size * rates
    factorial_vectorized = np.vectorize(np.math.factorial)
    const = -np.sum(np.log(factorial_vectorized(samples)))

    return term1 + term2 + const


class conditional_independence():

    def __init__(self):
        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): None,
            (0, 1): None,
            (1, 0): None,
            (1, 1): None
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): None,
            (0, 1): None,
            (1, 0): None,
            (1, 1): None
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): None,
            (0, 1): None,
            (1, 0): None,
            (1, 1): None
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): None,
            (0, 0, 1): None,
            (0, 1, 0): None,
            (0, 1, 1): None,
            (1, 0, 0): None,
            (1, 0, 1): None,
            (1, 1, 0): None,
            (1, 1, 1): None,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    coeff = 1.0 / (std * np.sqrt(2 * np.pi))
    exponent = -((x - mean) ** 2) / (2 * std ** 2)
    return coeff * np.exp(exponent)


class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        self.dataset = dataset
        self.class_value = class_value
        labels = dataset[:, -1]
        mask_c = (labels == class_value)
        data_c = dataset[mask_c]

        X_c = data_c[:, :-1]

        self.mean = np.mean(X_c, axis=0)
        self.std = np.std(X_c, axis=0, ddof=0)


    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        labels = self.dataset[:, -1]
        return np.sum(labels == self.class_value) / labels.shape[0]


    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        product = 1
        for mean, std, x_value in zip(self.mean, self.std, x):
            product *= normal_pdf(x_value, mean, std)

        return product

    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        return self.get_prior() * self.get_instance_likelihood(x)


class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1


    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        probability_0 = self.ccd0.get_instance_joint_prob(x)
        probability_1 = self.ccd1.get_instance_joint_prob(x)
        if probability_0 > probability_1:
            return 0
        return 1


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    d = mean.size

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    diff = x - mean
    exponent = -0.5 * diff.T.dot(inv_cov).dot(diff)
    norm_const = np.sqrt((2 * np.pi) ** d * det_cov)

    return np.exp(exponent) / norm_const


class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value
        labels = dataset[:, -1]
        mask_c = (labels == class_value)
        X_c = dataset[mask_c][:, :-1]
        self.mean = np.mean(X_c, axis=0)
        self.cov = np.cov(X_c, rowvar=False)

    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        labels = self.dataset[:, -1]
        return np.sum(labels == self.class_value) / labels.shape[0]

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        return self.get_prior() * self.get_instance_likelihood(x)


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    labels = test_set[:, -1]
    X = test_set[:, :-1]
    preds = np.array([map_classifier.predict(x) for x in X])
    return np.mean(preds == labels)


class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        self.dataset = dataset
        self.class_value = class_value
        # Separate data for this class
        labels = dataset[:, -1]
        mask = (labels == class_value)
        data_c = dataset[mask]
        X_c = data_c[:, :-1]
        self.n_j = X_c.shape[0]
        # Determine feature value sets V_t
        X_all = dataset[:, :-1]
        self.V = [np.unique(X_all[:, t]) for t in range(X_all.shape[1])]
        self.cond_probs = []
        for t, V_t in enumerate(self.V):
            counts = {v: 0 for v in V_t}
            for v in X_c[:, t]:
                counts[v] = counts.get(v, 0) + 1
            denom = self.n_j + len(V_t)
            probs = {v: (counts.get(v, 0) + 1) / denom for v in V_t}
            self.cond_probs.append(probs)

    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        labels = self.dataset[:, -1]
        return np.sum(labels == self.class_value) / labels.shape[0]

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        lik = 1.0
        for t, xi in enumerate(x):
            probs = self.cond_probs[t]
            # if unseen value, treatment: 1/(n_j+|V_t|)
            lik *= probs.get(xi, 1.0 / (self.n_j + len(self.V[t])))
        return lik

    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        return self.get_prior() * self.get_instance_likelihood(x)
