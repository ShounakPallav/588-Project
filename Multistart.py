import numpy as np  # type: ignore
from FDOptimization import runOptimization


def runCases():
    cases = [
        # np.array([-0.5, 0.5, 0.5]), #need to run
        np.array([0.5, 0.5, 0.5]),
        # np.array([0.5, 0.5, -0.5]), #need to run
    ]
    for case in cases:
        print(f'\n=== Multistart with kp {case[0]}, ki {case[1]}, kd {case[2]} ===')
        res = runOptimization(case)
        print(' status:', getattr(res, 'status', None))
        print(' message:', getattr(res, 'message', None))
        print(' fun:', getattr(res, 'fun', None))
        print(' x:', getattr(res, 'x', None))


if __name__ == "__main__":
    runCases()
