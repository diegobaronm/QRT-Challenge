import numpy as np
import pandas as pd
import sys

def checkOrthonormality(A): 
    bool = True
    D, F = A.shape   
    Error = pd.DataFrame(A.T @ A - np.eye(F)).abs()
    if any(Error.unstack() > 1e-6):
        bool = False
    return bool

# define a validation process for the generator
def validateA(generator):
    pass_orth_check = False
    retry = 0
    while pass_orth_check == False and retry < 5:
        A = generator()
        pass_orth_check = checkOrthonormality(A)
        retry += 1;
        if pass_orth_check:
            return A
        if pass_orth_check == False:
            print("A is not orthonormal, retrying...", file=sys.stderr)
        if retry == 5:
            raise ValueError("A is not orthonormal after 5 generations tries, please adjust the generator!")
        

# generators themselves, can not be called since the validation returns A
@validateA
def randomA(D=250, F=10):  
    M = np.random.randn(D,F)
    M = np.linalg.qr(M)[0]
    return M

@validateA
def momentomA():
    A = np.zeros((250,10))
    A[0:5, 0] = 1/np.sqrt(5) # 5-day return factor
    A[20:250, 1] = 1/np.sqrt(230) # momentum factor
    A = np.linalg.qr(A)[0]
    
    orthoProj = np.eye(250) - np.outer(A[:, 0], A[:, 0]) - np.outer(A[:, 1], A[:, 1]) - np.outer(A[:, 2], A[:, 2]) - np.outer(A[:, 3], A[:, 3]) # projection matrix on the orthogonal to the span of A[:,0] and A[:,1]
    A_remaining_columns = orthoProj @ np.random.randn(250, 6) # sample random vectors in the space orthogonal to the first two columns of A
    A_remaining_columns = np.linalg.qr(A_remaining_columns)[0] # orthonormalize these vectors with Gram-Schmidt algorithm
    A[:, 4:] = A_remaining_columns
    return A