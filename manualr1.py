import numpy as np
import itertools
import math

rates = np.array([[1, 1.98, 0.64, 1.34],
                  [0.48, 1, 0.31, 0.7],
                  [1.49, 3.1, 1, 1.95],
                  [0.72, 1.45, 0.52, 1]
                 ])
products = {0:'Shell', 1:'Pizza', 2:'Nugget', 3:'Snowball'}
def amount(seq):
    """Compute the final amount after a sequence of trades, starting with 1 SeaShell.

    Parameters
    ----------
    seq : list of int
        List of intermediate products traded.
    
    Returns
    -------
    float
        Payoff.
    """
    if not seq:
        return 1
    prod = 2000000*rates[0, seq[0]] * rates[seq[-1], 0]
    L = len(seq)
    for i in range(L-1):
        prod *= rates[seq[i], seq[i+1]]
    return prod
def maximize(L):
    """Among sequences of L intermediate products, compute the ones with greatest final amount.

    Parameters
    ----------
    L : int
        Number of intermediate products.
    
    Returns
    -------
    argmax : list of tuple
        Optimal sequences of intermediate trades.
    max_val : float
        Maximal final amount for L intermediate products.
    """
    seqs = itertools.product(*[range(0, 4) for _ in range(L)])
    max_val = float('-inf')
    argmax = []
    for seq in seqs:
        p = amount(seq)
        if math.isclose(p, max_val):
            argmax.append(seq)
        elif p > max_val:
            max_val = p
            argmax = [seq]
    return (argmax, max_val)
for L in range(0,5):
    print(maximize(L))



argmax, _ = maximize(4)
print("Optimal sequences of trades:")
for seq in argmax:
    res = ' -> '.join([products[0]] + [products[i] for i in seq] + [products[0]])
    print(res)

print(amount(()))