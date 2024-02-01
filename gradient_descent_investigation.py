import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.metrics.loss_functions import misclassification_error
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from itertools import product
from IMLearn.metrics import loss_functions

from utils import *
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection.cross_validate import cross_validate


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange]
        over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[
    1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the
    objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class,
        recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    """   
    following named arguments:
        - solver: GradientDescent
            self, the current instance of GradientDescent
        - weights: ndarray of shape specified by module's weights
            Current weights of objective
        - val: ndarray of shape specified by module's compute_output
        function
            Value of objective function at current point, over given
            data X, y
        - grad:  ndarray of shape specified by module's compute_jacobian
        function
            Module's jacobian with respect to the weights and at current
            point, over given data X,y
        - t: int
            Current GD iteration
        - eta: float
            Learning rate used at current iteration
        - delta: float
            Euclidean norm of w^(t)-w^(t-1)

    """

    values_lst, weights_lst = [], []

    def update_per_iteration(solver, weights, val, grad, t, eta, delta):
        values_lst.append(val)
        weights_lst.append(weights)

    return update_per_iteration, values_lst, weights_lst


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    modules_dict = {L1: "L1", L2: "L2"}
    for eta in etas:
        for l in [L1, L2]:
            l_module = l(weights=init.copy())
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta), out_type="best",
                                 callback=callback)
            module_fit = gd.fit(l_module)
            module_weights = np.vstack((init, np.array(weights)))
            module_values = np.array(values)

            plot_descent_path(l, module_weights
                              , title=f"GD Descent Path {modules_dict[l]} "
                                      f"norm with rate = {eta}",
                              xrange=[-1.5, 1.5],
                              yrange=[-1.5, 1.5]).show()
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=np.arange(len(module_values)) + 1,
                                        y=module_values, mode="lines"))
            figure.update_layout(
                title=f"GD Convergence Over {modules_dict[l]} norm using Rate "
                      f"eta="
                      f" {eta}")
            figure.show()
            print(
                f"{modules_dict[l]} module with rate {eta} gives minimum "
                f"loss: "
                f"{l(weights=module_fit).compute_output()}")


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    for gamma in gammas:
        l_module = L1(weights=init.copy())
        callback, values, weights = get_gd_state_recorder_callback()
        exp_lr = ExponentialLR(eta, gamma)
        gd = GradientDescent(exp_lr, out_type="best", callback=callback)
        output = gd.fit(l_module)
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=np.arange(len(values)) + 1,
                                    y=values, mode="markers+lines",
                                    showlegend=False, line=dict(
                color="purple")))
        figure.update_layout(
            title=f"Exponential learning rate : eta = {eta} , Gamma = {gamma}")
        figure.show()
        print(
            f"Exponential learning rate : eta = {eta} , Gamma = {gamma}"
            f"Lowest loss: "
            f"{L1(weights=output).compute_output():.5f}")

        if gamma == 0.95:
            gamma_weights = np.vstack((init, np.array(weights)))
    plot_descent_path(L1, gamma_weights,
                      title=f"Exponential learning rate: eta= {eta}, "
                            f"Gamma= 0.95",
                      xrange=[-1.5, 1.5], yrange=[-1.5, 1.5])


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train-
    and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples),
    n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples),
    n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    lambads = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train.to_numpy(), y_train.to_numpy())
    y_prob = logistic_regression.predict_proba(np.asarray(X_train))
    fpr, tpr, thresholds = roc_curve(np.asarray(y_train), y_prob)
    exp_argmax = np.argmax(tpr - fpr)
    best_alpha = np.round(thresholds[exp_argmax], 2)
    print(
        f"Best alpha: {best_alpha:}\nTest Error: "
        f"{logistic_regression.loss(X_test.to_numpy(), y_test.to_numpy())}")

    go.Figure(
        [go.Scatter(x=fpr, y=tpr, mode='lines')],
        layout=go.Layout(title=
                         f"$\text{{ROC Curve Of Fitted Logistic Regression "
                         f"Model - AUC}}={auc(fpr, tpr)}$",
                         xaxis=dict(
                             title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(
                             title=r"$\text{True Positive Rate ("
                                   r"TPR)}$"))).show()

    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation to specify values
    # of regularization parameter
    for l in ["l1", "l2"]:
        train_score_list, test_score_list = [], []
        for lam in lambads:
            logistic_regression = LogisticRegression(penalty=l, lam=lam)
            train_score, test_score = cross_validate(logistic_regression,
                                                     np.asarray(X_train),
                                                     np.asarray(y_train),
                                                     loss_functions.misclassification_error)
            train_score_list.append(train_score)
            test_score_list.append(test_score)
        best_lam = lambads[np.argmin(test_score_list)]
        logistic_regression_best_lam = LogisticRegression(penalty=l,
                                                          lam=best_lam).fit(
            np.asarray(X_train), np.asarray(y_train))
        test_error = logistic_regression_best_lam.loss(np.asarray(X_test),
                                                       np.asarray(y_test))
        print("the best lambda is: ", best_lam, "it's corresponding test "
                                                "error is: ",
              test_error)


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
