"""
Script that contains all the data generating processes
"""

# imports
import numpy as np
from scipy.special import expit
import pandas as pd

lb = 1
ub = 2

def fd_admg1(num_samples):
    """
    Generate data from an ADMG that satisfies the front door
    criterion.

    Z->A->M->Y; Z->M; A<->Y
    """

    # generate Z
    Z = np.random.binomial(1, 0.5, num_samples)

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,Z
    betaZM = np.random.uniform(lb, ub)
    betaAM = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-0.5 - betaZM*Z + betaAM*A), num_samples)

    # generate Y with random coeffs for Y|M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    Y = alphaU1Y*U1 + alphaU2Y*U2 - betaMY*M + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})

def fd_admg_nonlinear(num_samples):
    """
    Generate data from an ADMG that satisfies the front door
    criterion.

    Z->A->M->Y; Z->M; A<->Y
    """

    # generate Z
    Z = np.random.binomial(1, 0.5, num_samples)

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,Z
    betaZM = np.random.uniform(lb, ub)
    betaAM = np.random.uniform(lb, ub)
    betaAMZ = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-0.75 - betaZM*Z + betaAM*A + betaAMZ*Z*A), num_samples)

    # generate Y with random coeffs for Y|M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    Y = alphaU1Y*U1 + alphaU2Y*U2 - betaMY*M + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})

def fd_admg1_binary(num_samples):
    """
    Generate data from an ADMG that satisfies the front door
    criterion.

    Z->A->M->Y; Z->M; A<->Y
    """

    # generate Z
    Z = np.random.binomial(1, 0.5, num_samples)

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,Z
    betaZM = np.random.uniform(lb, ub)
    betaAM = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-0.5 - betaZM*Z + betaAM*A), num_samples)

    # generate Y with random coeffs for Y|M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    Y = np.random.binomial(1, expit(-0.5 + alphaU1Y*U1 + alphaU2Y*U2 - betaMY*M), num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})

def iv_admg1(num_samples):
    """
    Generate data from an ADMG that satisfies the front door
    criterion.

    Z->A->M->Y; A<->Y
    """

    # generate Z
    Z = np.random.binomial(1, 0.5, num_samples)

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,Z
    betaAM = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-1 + betaAM*A), num_samples)

    # generate Y with random coeffs for Y|M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    Y = alphaU1Y*U1 + alphaU2Y*U2 - betaMY*M + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})

def iv_admg_nonlinear(num_samples):
    """
    Generate data from an ADMG that satisfies the front door
    criterion.

    Z->A->M->Y; A<->Y
    """

    # generate Z
    Z = np.random.binomial(1, 0.5, num_samples)

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,Z
    betaAM = np.random.uniform(lb, ub)
    betaA2M = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-1 + betaAM*A), num_samples)

    # generate Y with random coeffs for Y|M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    betaUM = np.random.uniform(lb, ub)
    Y = alphaU1Y*U1 + alphaU2Y*U2 - betaMY*M**3 + betaUM*M*U1*U2 + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})

def fd_admg2(num_samples):
    """
    Generate data from an ADMG that satisfies the front door
    criterion.

    Z->A->M->Y; Z<->M; A<->Y
    """

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)
    U3 = np.random.uniform(-1, 1, num_samples)
    U4 = np.random.binomial(1, 0.5, num_samples)

    # generate Z with random coeffs for Z|U3,U4
    alphaU3Z = np.random.uniform(lb, ub)
    alphaU4Z = np.random.uniform(lb, ub)
    Z = np.random.binomial(1, expit(0.5 + alphaU3Z*U3 - alphaU4Z*U4), num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,U1,U2
    alphaU3M = np.random.uniform(lb, ub)
    alphaU4M = np.random.uniform(lb, ub)
    betaAM = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-0.5 - alphaU3M*U3 + alphaU4M*U4 + betaAM*A), num_samples)

    # generate Y with random coeffs for Y|M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    Y = alphaU1Y*U1 + alphaU2Y*U2 - betaMY*M + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})

def nonfd_admg1(num_samples):
    """
    Generate data from an ADMG that does not satisfy the front door
    criterion because of confounding between A-M; M-Y

    Z->A->M->Y; Z->M; A<->Y; A<->M, M<->Y
    """

    # generate Z
    Z = np.random.binomial(1, 0.5, num_samples)

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,Z,U1,U2
    betaZM = np.random.uniform(lb, ub)
    betaAM = np.random.uniform(lb, ub)
    alphaU1M = np.random.uniform(lb, ub)
    alphaU2M = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-0.5 - betaZM*Z + betaAM*A + alphaU1M*U1 - alphaU2M*U2), num_samples)

    # generate Y with random coeffs for Y|M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    Y = alphaU1Y*U1 + alphaU2Y*U2 - betaMY*M + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})

def nonfd_admg2(num_samples):
    """
    Generate data from an ADMG that does not satisfy the front door
    criterion because of A->Y direct effect

    Z->A->M->Y; Z->M; A<->Y; A->Y
    """

    # generate Z
    Z = np.random.binomial(1, 0.5, num_samples)

    # generate Us
    U1 = np.random.uniform(-1, 1, num_samples)
    U2 = np.random.binomial(1, 0.5, num_samples)

    # generate A with random coeffs for  A|Z,U1,U2
    betaZA = np.random.uniform(lb, ub)
    alphaU1A = np.random.uniform(lb, ub)
    alphaU2A = np.random.uniform(lb, ub)
    A = np.random.binomial(1, expit(-0.5 + betaZA*Z - alphaU1A*U1 + alphaU2A*U2), num_samples)

    # generate M with random coeffs for M|A,Z
    betaZM = np.random.uniform(lb, ub)
    betaAM = np.random.uniform(lb, ub)
    M = np.random.binomial(1, expit(-0.5 - betaZM*Z + betaAM*A), num_samples)

    # generate Y with random coeffs for Y|A,M,U1,U2
    alphaU1Y = np.random.uniform(lb, ub)
    alphaU2Y = np.random.uniform(lb, ub)
    betaMY = np.random.uniform(lb, ub)
    betaAY = np.random.uniform(lb, ub)
    Y = alphaU1Y*U1 + alphaU2Y*U2 - betaAY*A - betaMY*M + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({"Z": Z, "A": A, "M": M, "Y": Y, "U1": U1, "U2": U2})



if __name__ == "__main__":

    # test data from valid admgs
    data = fd_admg1(2000)
    print("FD ADMG1", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]))

    data = fd_admg2(2000)
    print("FD ADMG2", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]))

    # test data from an invalid admg
    data = nonfd_admg1(2000)
    print("Non FD ADMG 1", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]))

    data = nonfd_admg2(2000)
    print("Non FD ADMG2", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]))

    data = fd_admg1_binary(2000)
    print("FD ADMG 1 binary Y", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]), np.mean(data["Y"]))

    data = iv_admg1(2000)
    print("IV ADMG 1", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]))

    data = iv_admg_nonlinear(2000)
    print("IV ADMG non linear", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]))

    data = fd_admg_nonlinear(2000)
    print("FD ADMG non linear", np.mean(data["Z"]), np.mean(data["A"]), np.mean(data["M"]))
