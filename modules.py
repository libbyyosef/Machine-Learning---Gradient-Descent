import numpy as np
from IMLearn import BaseModule


class L2(BaseModule):
    """
    Class representing the L2 module

    Represents the function: f(w)=||w||^2_2
    """

    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L2 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        w=self.weights
        w_norm_2=np.linalg.norm(w,ord=2)
        normalize_w_pow_2=np.power(w_norm_2,2)
        return normalize_w_pow_2

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L2 derivative with respect to self.weights at point
        self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L2 derivative with respect to self.weights at point self.weights
        """
        w=self.weights
        double_w=2*w
        return double_w


class L1(BaseModule):
    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the L1 function at point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        w=self.weights
        w_norm_1=np.linalg.norm(w, ord=1)
        return w_norm_1

    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute L1 derivative with respect to self.weights at point
        self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            L1 derivative with respect to self.weights at point self.weights
        """
        w=self.weights
        w_sign=np.sign(w)
        return w_sign



class LogisticModule(BaseModule):
    """
    Class representing the logistic regression objective function

    Represents the function: f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(
    <x_i,w>))]
    """

    def __init__(self, weights: np.ndarray = None):
        """
        Initialize a logistic regression module instance

        Parameters:
        ----------
        weights: np.ndarray, default=None
            Initial value of weights
        """
        super().__init__(weights)

    def compute_output(self, X: np.ndarray, y: np.ndarray,
                       **kwargs) -> np.ndarray:
        """
        Compute the output value of the logistic regression objective
        function at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        # f(w) = - (1/m) sum_i^m[y*<x_i,w> - log(1+exp(<x_i,w>))]

        w=self.weights
        m=X.shape[0]
        X_w_product=X@w
        y_X_w_mult=y*X_w_product
        exp_expression=np.exp(X_w_product)
        log_expression=np.log(1+exp_expression)
        objective_sum=np.sum(y_X_w_mult-log_expression)
        res=(-1)*objective_sum/m
        return res

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray,
                         **kwargs) -> np.ndarray:
        """
        Compute the gradient of the logistic regression objective function
        at point self.weights

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Design matrix to use when computing objective

        y: ndarray of shape(n_samples,)
            Binary labels of samples to use when computing objective

        Returns
        -------
        output: ndarray of shape (n_features,)
            Derivative of function with respect to self.weights at point
            self.weights
        """
        w=self.weights
        m=X.shape[0]
        X_w_product=X@w
        exp_X_w_product=np.exp(X_w_product)
        diff_term = y[:, np.newaxis] * X - (exp_X_w_product / (1 + exp_X_w_product))[:,
                                           np.newaxis] * X
        jacobian = -np.sum(diff_term, axis=0) / m
        return jacobian




class RegularizedModule(BaseModule):
    """
    Class representing a general regularized objective function of the format:
                                    f(w) = F(w) + lambda*R(w)
    for F(w) being some fidelity function, R(w) some regularization function
    and lambda
    the regularization parameter
    """

    def __init__(self,
                 fidelity_module: BaseModule,
                 regularization_module: BaseModule,
                 lam: float = 1.,
                 weights: np.ndarray = None,
                 include_intercept: bool = True):
        """
        Initialize a regularized objective module instance

        Parameters:
        -----------
        fidelity_module: BaseModule
            Module to be used as a fidelity term

        regularization_module: BaseModule
            Module to be used as a regularization term

        lam: float, default=1
            Value of regularization parameter

        weights: np.ndarray, default=None
            Initial value of weights

        include_intercept: bool default=True
            Should fidelity term (and not regularization term) include an
            intercept or not
        """
        super().__init__()
        self.fidelity_module_, self.regularization_module_, self.lam_ = \
            fidelity_module, regularization_module, lam
        self.include_intercept_ = include_intercept

        if weights is not None:
            self.weights = weights

    def compute_output(self, **kwargs) -> np.ndarray:
        """
        Compute the output value of the regularized objective function at
        point self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (1,)
            Value of function at point self.weights
        """
        F_w = self.fidelity_module_.compute_output(**kwargs)
        R_w = self.regularization_module_.compute_output(**kwargs)
        lambda_R_w = self.lam_ * R_w
        return F_w + lambda_R_w


    def compute_jacobian(self, **kwargs) -> np.ndarray:
        """
        Compute module derivative with respect to self.weights at point
        self.weights

        Parameters
        ----------
        kwargs:
            No additional arguments are expected

        Returns
        -------
        output: ndarray of shape (n_in,)
            Derivative with respect to self.weights at point self.weights
        """
        regularization_jacobian = self.regularization_module_.compute_jacobian(
            **kwargs)
        fidelity_jacobian = self.fidelity_module_.compute_jacobian(**kwargs)
        lambda_val=self.lam_
        regularization_jacobian = np.insert(
            regularization_jacobian, 0,
            0) if self.include_intercept_ else regularization_jacobian
        lambda_regularization_jacobian = lambda_val * regularization_jacobian
        output = fidelity_jacobian + lambda_regularization_jacobian
        return output


    @property
    def weights(self):
        """
        Wrapper property to retrieve module parameter

        Returns
        -------
        weights: ndarray of shape (n_in, n_out)
        """
        return self.weights_

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        """
        Setter function for module parameters

        In case self.include_intercept_ is set to True, weights[0] is
        regarded as the intercept
        and is not passed to the regularization module

        Parameters
        ----------
        weights: ndarray of shape (n_in, n_out)
            Weights to set for module
        """
        self.weights_ = weights
        self.regularization_module_.weights=weights[1:] if \
            self.include_intercept_ else weights
        self.fidelity_module_.weights = weights


