import numpy as np


class Regressor(object):
    def score(self, X, y):
        meany = np.mean(y)
        SStot = 0
        SSres = 0
        for xi, yi in zip(X, y):
            res = yi - self.predict(xi)
            tot = yi - meany
            SSres += res*res
            SStot += tot*tot
        return 1 - SSres/SStot
