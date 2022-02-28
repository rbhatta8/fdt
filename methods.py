"""
Script that implements all the Verma tests and causal effect estimation
"""

# imports
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from dgps import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from fcit import fcit


def parametric_gnull_weights(data, M, mpM):
    """
    Compute usual verma weights p(M|mpM)

    M: name of variable M (assumed to be binary in our code)
    mpM: list of variable names for Markov pillow of M
    """

    # fit a model for M
    formula_M = M + "~ 1"
    if len(mpM) > 0:
        formula_M += " + " + "+".join(mpM)
    modelM = sm.GLM.from_formula(formula=formula_M, data=data, family=sm.families.Binomial()).fit()

    # use modelM to predict propensity scores for Verma test
    p_scores = modelM.predict(data)
    p_scores[data[M]==0] = 1 - p_scores[data[M]==0]

    # return weights
    return 1/p_scores

def parametric_dual_weights(data, M, mpM, A, Avalue=1):
    """
    Compute dual weights p(M|mpM)/p(M|mpM) evaluated at A=1 in numerator.

    M: name of variable M (assumed to be binary in our code)
    mpM: list of variable names for Markov pillow of M
    A: name of treatment variable
    Avalue: intervention value for A
    """

    # fit a model for M
    formula_M = M + "~ 1"
    if len(mpM) > 0:
        formula_M += " + " + "+".join(mpM)
    modelM = sm.GLM.from_formula(formula=formula_M, data=data, family=sm.families.Binomial()).fit()

    # use modelM to predict propensity scores for Verma test
    p_scores = modelM.predict(data)
    p_scores[data[M]==0] = 1 - p_scores[data[M]==0]

    # get data sets where we assign A to 1 to stabilize weights
    dataAfix = data.copy()
    dataAfix[A] = Avalue
    p_scoresAfix = modelM.predict(dataAfix)
    p_scoresAfix[data[M]==0] = 1 - p_scoresAfix[data[M]==0]

    # return weights
    return p_scoresAfix/p_scores

def ml_gnull_weights(data, M, mpM):
    """
    Compute gnull weights p(M|mpM) using a random forest.

    M: name of variable M (assumed to be binary in our code)
    mpM: list of variable names for Markov pillow of M
    """

    # get data in the right shape
    Mvec = data[M].to_numpy().reshape(len(data),)
    Mpredictors = data[mpM].to_numpy()

    # fit random forest for M
    model_M = RandomForestClassifier(bootstrap=False)
    model_M.fit(Mpredictors, Mvec)

    # get predictions for M
    p_scores = model_M.predict_proba(Mpredictors)[:,1]
    p_scores[data[M]==0] = 1 - p_scores[data[M]==0]

    return 1/p_scores

def ml_dual_weights(data, M, mpM, A, Avalue=1):
    """
    Compute dual weights p(M|mpM)/p(M|mpM) evaluated at A=1 in numerator
    using a random forest.

    M: name of variable M (assumed to be binary in our code)
    mpM: list of variable names for Markov pillow of M
    """

    # get data in the right shape
    Mvec = data[M].to_numpy().reshape(len(data),)
    Mpredictors = data[mpM].to_numpy()
    dataAfix = data.copy()
    dataAfix[A] = Avalue
    MpredictorsAfix = dataAfix[mpM].to_numpy()

    # fit random forest for M
    model_M = RandomForestClassifier(bootstrap=False)
    model_M.fit(Mpredictors, Mvec)

    # get predictions for M
    p_scores = model_M.predict_proba(Mpredictors)[:,1]
    p_scores[data[M]==0] = 1 - p_scores[data[M]==0]
    p_scoresAfix = model_M.predict_proba(MpredictorsAfix)[:,1]
    p_scoresAfix[data[M]==0] = 1 - p_scoresAfix[data[M]==0]

    return p_scoresAfix/p_scores


def _get_probY(data, Y, modelY):
    """
    Helper function to get probability of continuous Y
    """

    E_Y = modelY.predict(data)
    std = np.std(data[Y] - E_Y)
    return  stats.norm.pdf(data[Y], loc=E_Y, scale=std)


def parametric_primal_weights(data, A, Y, mpA, mpY):
    """
   Compute primal weights \sum_A p(A|mpA)p(Y|mpY) / p(A|mpA)p(Y|mpY)

    Y: name of outcome variable (assumed to be a continuous variable)
    A: name of treatment variable (assumed to be binary)
    mpA: list of variables names for Markov pillow of A
    mpY: list of variable names for Markov pillow of Y
    """

    formula_A = A + "~ 1 + " + "+".join(mpA)
    formula_Y = Y + "~1 + " + "+".join(mpY)

    # fit models for A and Y
    model_A = sm.GLM.from_formula(formula=formula_A, data=data, family=sm.families.Binomial()).fit()
    model_Y = sm.GLM.from_formula(formula=formula_Y, data=data, family=sm.families.Gaussian()).fit()

    # get predictions for A
    pA = model_A.predict(data)

    # get data sets where we assign A to 1 and 0
    dataA1 = data.copy(); dataA0 = data.copy()
    dataA1[A] = 1; dataA0[A] = 0

    # get pY, pYA0 and pYA1
    pY = _get_probY(data, Y, model_Y)
    pYA0 = _get_probY(dataA0, Y, model_Y)
    pYA1 = _get_probY(dataA1, Y, model_Y)

    # compute numerator \sum_A p(A|mpA)p(Y|mpY)
    numerator = (1-pA)*pYA0 + pA*pYA1

    # compute denominator p(A|mpA)p(Y|mpY)
    pA = pA*data[A] + (1-pA)*(1-data[A])
    denominator = pA*pY

    return numerator/denominator

def ml_primal_weights_binary(data, A, Y, mpA, mpY):
    """
   Compute primal weights \sum_A p(A|mpA)p(Y|mpY) / p(A|mpA)p(Y|mpY)

    A: name of treatment variable (assumed to be binary)
    Y: name of outcome variable (assumed to be a binary variable)
    mpA: list of variables names for Markov pillow of A
    mpY: list of variable names for Markov pillow of Y
    """

    Avec = data[A].to_numpy().reshape(len(data),)
    Yvec = data[Y].to_numpy().reshape(len(data),)
    Apredictors = data[mpA].to_numpy()
    Ypredictors = data[mpY].to_numpy()

    # fit models for A and Y
    model_A = RandomForestClassifier(bootstrap=False)
    model_A.fit(Apredictors, Avec)
    model_Y = RandomForestClassifier(bootstrap=False)
    model_Y.fit(Ypredictors, Yvec)

    # get predictions for A
    pA = model_A.predict_proba(Apredictors)[:,1]

    # get data sets where we assign A to 1 and 0
    dataA1 = data.copy(); dataA0 = data.copy()
    dataA1[A] = 1; dataA0[A] = 0
    YpredictorsA1 = dataA1[mpY].to_numpy(); YpredictorsA0 = dataA0[mpY].to_numpy()

    # get pY, pYA0 and pYA1
    pY = model_Y.predict_proba(Ypredictors)[:, 1]
    pY[data[Y] == 0] = 1 - pY[data[Y] == 0]
    pYA0 = model_Y.predict_proba(YpredictorsA0)[:, 1]
    pYA0[data[Y] == 0] = 1 - pYA0[data[Y] == 0]
    pYA1 = model_Y.predict_proba(YpredictorsA1)[:, 1]
    pYA1[data[Y] == 0] = 1 - pYA1[data[Y] == 0]

    # compute numerator \sum_A p(A|mpA)p(Y|mpY)
    numerator = (1-pA)*pYA0 + pA*pYA1

    # compute denominator
    pA = pA*data[A] + (1-pA)*(1-data[A])
    denominator = pA*pY

    return numerator/denominator

def parametric_primal_weights_binary(data, A, Y, mpA, mpY):
    """
   Compute primal weights \sum_A p(A|mpA)p(Y|mpY) / p(A|mpA)p(Y|mpY)

    A: name of treatment variable (assumed to be binary)
    Y: name of outcome variable (assumed to be a binary variable)
    mpA: list of variables names for Markov pillow of A
    mpY: list of variable names for Markov pillow of Y
    """

    formula_A = A + "~ 1 + " + "+".join(mpA)
    formula_Y = Y + "~1 + " + "+".join(mpY)

    # fit models for A and Y
    model_A = sm.GLM.from_formula(formula=formula_A, data=data, family=sm.families.Binomial()).fit()
    model_Y = sm.GLM.from_formula(formula=formula_Y, data=data, family=sm.families.Binomial()).fit()

    # get predictions for A
    pA = model_A.predict(data)

    # get data sets where we assign A to 1 and 0
    dataA1 = data.copy(); dataA0 = data.copy()
    dataA1[A] = 1; dataA0[A] = 0

    # get pY, pYA0 and pYA1
    pY = model_Y.predict(data)
    pY[data[Y] == 0] = 1 - pY[data[Y] == 0]
    pYA0 = model_Y.predict(dataA0)
    pYA0[data[Y] == 0] = 1 - pYA0[data[Y] == 0]
    pYA1 = model_Y.predict(dataA1)
    pYA1[data[Y] == 0] = 1 - pYA1[data[Y] == 0]

    # compute numerator \sum_A p(A|mpA)p(Y|mpY)
    numerator = (1-pA)*pYA0 + pA*pYA1

    # compute denominator
    pA = pA*data[A] + (1-pA)*(1-data[A])
    denominator = pA*pY

    return numerator/denominator

def weighted_lr_test(data, Y, Z, cond_set=[], weights=None, state_space="continuous"):
    """
    Perform a weighted likelihood ratio test for a Verma constraint Y _||_ Z | cond_set in a reweighted distribution

    Y: name of outcome variable (assumed to be a continuous variable)
    Z: name of anchor variable
    cond_set: list of variable names that are conditioned on when checking independence
    state_space: "continuous" if Y is a continuous variable, otherwise Y is treated as binary
    """

    # fit weighted null and alternative models to check independence in the kernel
    formula_Y = Y + "~ 1"
    if len(cond_set) > 0:
        formula_Y += " + " + "+".join(cond_set)

    if weights is None:
        weights = np.ones(len(data))

    if state_space == "continuous":
        modelY_null = sm.GLM.from_formula(formula=formula_Y, data=data, freq_weights=weights, family=sm.families.Gaussian()).fit()
        modelY_alt = sm.GLM.from_formula(formula=formula_Y + "+" + Z, data=data, freq_weights=weights, family=sm.families.Gaussian()).fit()
    elif state_space == "binary":
        modelY_null = sm.GLM.from_formula(formula=formula_Y, data=data, freq_weights=weights, family=sm.families.Binomial()).fit()
        modelY_alt = sm.GLM.from_formula(formula=formula_Y + "+" + Z, data=data, freq_weights=weights, family=sm.families.Binomial()).fit()
    else:
        print("Invalid state space for outcome.")
        assert(False)

    # the test statistic 2*(loglike_alt - loglike_null) is chi2 distributed
    chi2_stat = 2*(modelY_alt.llf - modelY_null.llf)
    return 1 - stats.chi2.cdf(x=chi2_stat, df=1)


def fcit_test(data, Y, Z, cond_set=[], weights=None):
    """
    FCIT test
    """

    # resample the data with replacement and with weights
    data_sampled = data.sample(int(len(data)/2), replace=True, weights=weights)
    data_sampled.reset_index(drop=True, inplace=True)

    yvec = data_sampled[[Y]].to_numpy()
    zvec = data_sampled[[Z]].to_numpy()

    if len(cond_set) > 0:
        cvec = data_sampled[cond_set].to_numpy()
        return fcit.test(yvec, zvec, cvec)
    return fcit.test(yvec, zvec)

def primal_ipw(data, Y, A, weights):
    """
    Compute causal effect using primal IPW

    Y: name of outcome variable (assumed to be a continuous variable)
    A: name of treatment variable (assumed to be binary)
    weights: pre-computed primal weights
    """

    return np.mean(data[A]*data[Y]*weights - (1-data[A])*data[Y]*weights)

def dual_ipw(data, Y, weightsA0, weightsA1):
    """
    Compute dual weights p(M|mpM)

    Y: name of outcome variable
    weightsA0: pre-computed dual weights when A=0
    weightsA1: pre-computed dual weights when A=1
    """

    # return the effect
    return np.mean(data[Y]*weightsA1 - data[Y]*weightsA0)

def parametric_iv(data, Y, A, Z, cond_set=[], state_space="continuous"):
    """
    Compute parametric IV estimates
    """

    formulaY = Y + "~" + Z
    if len(cond_set) > 0:
        formulaY += "+" + "+".join(cond_set)

    formulaA = A + "~" + Z
    if len(cond_set) > 0:
        formulaA += "+" + "+".join(cond_set)

    if state_space == "continuous":
        modelY = sm.GLM.from_formula(formula=formulaY, data=data, family=sm.families.Gaussian()).fit()
    elif state_space == "binary":
        modelY = sm.GLM.from_formula(formula=formulaY, data=data, family=sm.families.Binomial()).fit()
    else:
        print("Unkown state space")
        assert(False)

    modelA = sm.GLM.from_formula(formula=formulaA, data=data, family=sm.families.Binomial()).fit()

    dataZ0 = data.copy(); dataZ1 = data.copy()
    dataZ0[Z] = 0; dataZ1[Z] = 1
    numerator = np.mean(modelY.predict(dataZ1)) - np.mean(modelY.predict(dataZ0))
    denominator = np.mean(modelA.predict(dataZ1)) - np.mean(modelA.predict(dataZ0))
    return numerator/denominator

def parametric_backdoor(data, Y, A, C, state_space="continuous"):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment

    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    C: list of variable names to be included in backdoor adjustment set
    """

    formula = Y + "~" + A
    if len(C) > 0:
        formula += " + " + "+".join(C)

    if state_space == "continuous":
        model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    else:
        model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Binomial()).fit()
    data_A0 = data.copy()
    data_A1 = data.copy()
    data_A0[A] = 0
    data_A1[A] = 1
    return(np.mean(model.predict(data_A1)) - np.mean(model.predict(data_A0)))

def ml_backdoor(data, Y, A, C, state_space="continuous"):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment

    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    C: list of variable names to be included in backdoor adjustment set
    """

    Yvec = data[Y].to_numpy().reshape(len(data),)
    Ypredictors = data[[A] + C].to_numpy()

    # fit model for Y
    model_Y = RandomForestRegressor(bootstrap=False)
    model_Y.fit(Ypredictors, Yvec)

    data_A0 = data.copy(); data_A1 = data.copy()
    data_A0[A] = 0; data_A1[A] = 1
    YpredictorsA1 = data_A1[[A] + C].to_numpy()
    YpredictorsA0 = data_A0[[A] + C].to_numpy()
    return(np.mean(model_Y.predict(YpredictorsA1)) - np.mean(model_Y.predict(YpredictorsA0)))


if __name__ == "__main__":

    num_samples = 5000
    # try the dgps that satisfy front-door
    data = fd_admg1(num_samples)
    dual_verma_weights_A0 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 0)
    dual_verma_weights_A1 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 1)
    dual_p_val = weighted_lr_test(data, "Y", "Z", ["M"], dual_verma_weights_A1)
    primal_weights = parametric_primal_weights(data, "A", "Y", ["Z"], ["Z", "A", "M"])
    primal_p_val = weighted_lr_test(data, "Y", "Z", ["M"], primal_weights)
    print("FD DGP 1", dual_p_val, primal_p_val, "dual, primal p-vals should be > 0.05")
    print("Parametric backdoor with U", parametric_backdoor(data, "Y", "A", ["U1", "U2", "Z"]))
    print("Parametric dual IPW", dual_ipw(data, "Y", dual_verma_weights_A0, dual_verma_weights_A1))
    print("Parametric primal IPW", primal_ipw(data, "Y", "A", primal_weights))
    print("-"*5)

    data = fd_admg2(num_samples)
    dual_verma_weights_A0 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 0)
    dual_verma_weights_A1 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 1)
    dual_p_val = weighted_lr_test(data, "Y", "Z", ["M"], dual_verma_weights_A1)
    primal_weights = parametric_primal_weights(data, "A", "Y", ["Z"], ["Z", "A", "M"])
    primal_p_val = weighted_lr_test(data, "Y", "Z", ["M"], primal_weights)
    print("FD DGP 2", dual_p_val, primal_p_val, "dual, primal p-vals should be > 0.05")
    print("Parametric backdoor with U", parametric_backdoor(data, "Y", "A", ["U1", "U2", "Z"]))
    print("Parametric dual IPW", dual_ipw(data, "Y", dual_verma_weights_A0, dual_verma_weights_A1))
    print("Parametric primal IPW", primal_ipw(data, "Y", "A", primal_weights))
    print("-"*5)

    # try the dgps that don't satisfy front-door
    data = nonfd_admg2(num_samples)
    dual_verma_weights_A0 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 0)
    dual_verma_weights_A1 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 1)
    dual_p_val = weighted_lr_test(data, "Y", "Z", ["M"], dual_verma_weights_A1)
    primal_weights = parametric_primal_weights(data, "A", "Y", ["Z"], ["Z", "A", "M"])
    ml_primal_weights = parametric_primal_weights(data, "A", "Y", ["Z"], ["Z", "A", "M"])
    primal_p_val = weighted_lr_test(data, "Y", "Z", ["M"], primal_weights)
    print("Non FD DGP 1", dual_p_val, primal_p_val, "dual, primal p-vals should be < 0.05")
    print("Parametric backdoor with U", parametric_backdoor(data, "Y", "A", ["U1", "U2", "Z"]))
    print("Parametric dual IPW", dual_ipw(data, "Y", dual_verma_weights_A0, dual_verma_weights_A1))
    print("Parametric primal IPW", primal_ipw(data, "Y", "A", primal_weights))
    print("-"*5)

    data = nonfd_admg2(num_samples)
    dual_verma_weights_A0 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 0)
    dual_verma_weights_A1 = parametric_dual_weights(data, "M", ["Z", "A"], "A", 1)
    dual_p_val = weighted_lr_test(data, "Y", "Z", ["M"], dual_verma_weights_A1)
    primal_weights = parametric_primal_weights(data, "A", "Y", ["Z"], ["Z", "A", "M"])
    primal_p_val = weighted_lr_test(data, "Y", "Z", ["M"], primal_weights)
    print("Non FD DGP 2", dual_p_val, primal_p_val, "dual, primal p-vals should be < 0.05")
    print("Parametric backdoor with U", parametric_backdoor(data, "Y", "A", ["U1", "U2", "Z"]))
    print("Parametric dual IPW", dual_ipw(data, "Y", dual_verma_weights_A0, dual_verma_weights_A1))
    print("Parametric primal IPW", primal_ipw(data, "Y", "A", primal_weights))
    print("-"*5)

    data = iv_admg1(num_samples + 2000)
    ml_dual_weights_A0 = ml_dual_weights(data, "M", ["Z", "A"], "A", 0)
    ml_dual_weights_A1 = ml_dual_weights(data, "M", ["Z", "A"], "A", 1)
    ml_dual_p_val = fcit_test(data, "Y", "Z", ["M"], ml_dual_weights_A1)
    print("Parametric backdoor with U", parametric_backdoor(data, "Y", "A", ["U1", "U2", "Z"]))
    print("ML dual IPW", dual_ipw(data, "Y", ml_dual_weights_A0, ml_dual_weights_A1))
