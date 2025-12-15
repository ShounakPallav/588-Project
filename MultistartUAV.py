import numpy as np  # type: ignore
from FDOptUAV import runOptimization

def runCases():
    cases = [
        # np.array([0.04, 0.01, 2.0, 0.6, 0.25, 0.3, 0.08]), # ran already
        # np.array([0.26, 0.0058, 3.297, 0.730, -1.417, 8.234, 2.245]), # ran already
        # np.array([0.1211, 0.0058, 2.176, 0.6, 0.1, 1.417, 2.410]) # ran already
    ]
    for case in cases:
        print(f'\n=== Multistart UAV with k0={case} ===')
        res = runOptimization(case)
        print(' status:', getattr(res, 'status', None))
        print(' message:', getattr(res, 'message', None))
        print(' fun:', getattr(res, 'fun', None))
        print(' x:', getattr(res, 'x', None))


if __name__ == "__main__":
    runCases()
