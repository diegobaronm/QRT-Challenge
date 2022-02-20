import numpy as np
import pandas as pd
import lib.QRTLib as qrtl
import lib.genALib as gal
import lib.fitBetaLib as fbl

def main():
    x_train, y_train = qrtl.dataPreProc()
    A = gal.momentomA
    beta = fbl.fitBetaLinear(A, x_train, y_train)
    y_pred = qrtl.getYPred(x_train, A, beta)
    metric = qrtl.metric(y_train, y_pred)
    qrtl.outputCSV(A, beta, surfix= 'momA_linB_' + str(round(metric, 5)))

if __name__ == '__main__':
    main()


