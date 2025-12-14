import os
import time
import csv
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matlab.engine  # type: ignore
from scipy.optimize import minimize, Bounds  # type: ignore

# MATLAB setup (shared engine)
eng = matlab.engine.connect_matlab('engine_1')
model = 'ProofofConceptProblem'
eng.load_system(model, nargout=0)
eng.set_param(model, 'SimulationCommand', 'stop', nargout=0)
eng.set_param(model, 'FastRestart', 'off', nargout=0)
eng.set_param(model, 'SolverType', 'Variable-step', nargout=0)
eng.set_param(model, 'Solver', 'ode45', nargout=0)
eng.set_param(model, 'StopTime', '25', nargout=0)
eng.set_param(model, 'RelTol', '1e-6', nargout=0)
eng.set_param(model, 'AbsTol', '1e-8', nargout=0)
eng.set_param(model, 'MaxStep', '1e-3', nargout=0)


def fmtVal(val):
    return f"{val:.6g}".replace('+', '').replace(' ', '')


def buildLabel(k):
    return f"kp{fmtVal(k[0])}_ki{fmtVal(k[1])}_kd{fmtVal(k[2])}"


def toVec(x):
    return np.asarray(x).squeeze()


def tailValue(y, n=20):
    n = min(n, y.size)
    return np.mean(y[-n:])


def settle(t, y, yss, pct=0.02):
    band = pct * abs(yss) if yss != 0 else pct
    err = np.abs(y - yss)
    ok = err <= band
    stay = ok.copy()
    for i in range(len(stay) - 2, -1, -1):
        stay[i] = stay[i] and stay[i + 1]
    i0 = np.argmax(stay)
    if not stay[i0]:
        return np.inf
    j = i0
    while j > 0 and err[j - 1] <= band:
        j -= 1
    if j == 0:
        return float(t[i0])
    e1 = err[j - 1] - band
    e2 = err[j] - band
    if e1 == e2:
        return float(t[j])
    alpha = -e1 / (e2 - e1)
    tcross = t[j - 1] + alpha * (t[j] - t[j - 1])
    return float(tcross)


def feasibilityResidual(k, m, sse):
    viol = 0.0
    viol = max(viol, m - 0.10)
    viol = max(viol, abs(sse) - 1e-3)
    for ki, bound in zip(k, (-10.0, -10.0, -10.0)):
        viol = max(viol, bound - ki)
    for ki, bound in zip(k, (10.0, 10.0, 10.0)):
        viol = max(viol, ki - bound)
    return viol


def runOptimization(initK, baseDir=None):
    label = buildLabel(initK)
    outDir = baseDir if baseDir else f"slices_{label}"
    os.makedirs(outDir, exist_ok=True)
    logPath = os.path.join(outDir, f"opt_log_{label}.csv")

    startTime = time.time()
    simCount = {'n': 0}
    evalCount = {'n': 0}
    lastLogged = {'idx': 0, 'k': None}

    with open(logPath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SLSQPiter', 'kp', 'ki', 'kd', 'Ts', 'M', 'SSe', 'feas', 'feasResidual', 'Time(s)', 'gradnorm', 'funCalls'])

    def saferun(k, ymax=1e3, count=False):
        try:
            kp, ki, kd = float(k[0]), float(k[1]), float(k[2])
            eng.workspace['kp'] = kp
            eng.workspace['ki'] = ki
            eng.workspace['kd'] = kd
            eng.sim(model, nargout=0)
            if count:
                simCount['n'] += 1
            t = toVec(eng.workspace['time'])
            y = toVec(eng.workspace['y'])
            r = toVec(eng.workspace['r'])
        except Exception:
            return None, None, None, np.inf, np.inf, np.inf, False

        if t.size == 0 or y.size == 0 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
            return t, y, r, np.inf, np.inf, np.inf, False
        if np.max(np.abs(y)) > ymax:
            return t, y, r, np.inf, np.inf, np.inf, False

        yss = tailValue(y, max(10, y.size // 10))
        rss = tailValue(r, max(10, r.size // 10)) if r.size > 1 else float(r)
        m = (np.max(y) - yss) / (yss if yss != 0 else 1.0)
        ts = settle(t, y, yss, pct=0.02)
        sse = rss - yss

        if not np.isfinite(ts):
            return t, y, r, np.inf, max(m, 0.11), np.inf, False

        return t, y, r, ts, m, sse, True

    def once(k, count=False):
        t, y, r, ts, m, sse, ok = saferun(k, count=count)
        return t, y, r, ts, m, sse, ok

    def evalCostSilent(k):
        _, _, _, ts, m, sse, ok = once(k, count=False)
        if not ok or not np.isfinite(ts):
            return np.inf
        return ts

    def gradNormEst(k, f0=None, h0=1e-2):
        k = np.asarray(k, float)
        g = np.zeros_like(k)
        fBase = evalCostSilent(k) if f0 is None else f0
        for i in range(len(k)):
            step = max(h0, 0.05 * max(1.0, abs(k[i])))
            e = np.zeros_like(k)
            e[i] = 1.0
            fp = evalCostSilent(k + step * e)
            fm = evalCostSilent(k - step * e)
            if np.isfinite(fp) and np.isfinite(fm):
                g[i] = (fp - fm) / (2 * step)
            elif np.isfinite(fp):
                g[i] = (fp - fBase) / step
            elif np.isfinite(fm):
                g[i] = (fBase - fm) / step
            else:
                g[i] = 0.0
        return float(np.linalg.norm(g))

    def logRow(iterIdx, k, ts, m, sse, feas, feasRes, elapsed, gn):
        with open(logPath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iterIdx, float(k[0]), float(k[1]), float(k[2]), ts, m, sse, feas, feasRes, elapsed, gn, simCount['n']])
        lastLogged['idx'] = iterIdx
        lastLogged['k'] = np.asarray(k, float)

    def savePlot(k, idx=None, data=None):
        if data is None:
            t, y, r, ts, m, sse, ok = once(k, count=False)
        else:
            t, y, r, ts, m, sse, ok = data
        plt.figure()
        ssVal = None
        if t is not None and y is not None:
            ssVal = tailValue(y, max(10, y.size // 10))
            plt.plot(t, y, label='y')
            if r.size == y.size:
                plt.plot(t, r, '--', label='r')
            else:
                val = float(r) if r.size == 1 else tailValue(r)
                plt.axhline(val, linestyle='--', label='r')
            if ssVal is not None:
                plt.axhline(0.98 * ssVal, color='gray', linestyle='--', linewidth=1.0, label='0.98*y_ss')
                plt.axhline(1.02 * ssVal, color='gray', linestyle='--', linewidth=1.0, label='1.02*y_ss')
                plt.axhline(1.10 * ssVal, color='red', linestyle='--', linewidth=1.0, label='OS 10% bound')
        ttl = f'kp={k[0]:.3g} ki={k[1]:.3g} kd={k[2]:.3g}  Ts={ts:.3g}  M={m:.3g}  SSe={sse:.2e}'
        plt.title(ttl)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plotIdx = lastLogged['idx'] if idx is None else idx
        name = os.path.join(outDir, f'iter{plotIdx:03d}_{label}.pdf')
        plt.savefig(name)
        plt.close()

    def fun(k):
        _, _, _, ts, m, sse, ok = once(k, count=True)
        evalCount['n'] += 1
        if not ok:
            print(f'FDeval {evalCount["n"]:04d}: kp {k[0]:.16g} ki {k[1]:.16g} kd {k[2]:.16g}  unstable or bad -> large cost')
            return 1e6
        print(f'FDeval {evalCount["n"]:04d}: kp {k[0]:.16g} ki {k[1]:.16g} kd {k[2]:.16g}  Ts {ts:.16g}  M {m:.16g}  SSe {sse:.16g}')
        return ts

    def consVals(k):
        _, _, _, ts, m, sse, ok = once(k, count=True)
        if not ok or not np.isfinite(m) or not np.isfinite(sse):
            return np.array([-1.0, -1.0, -1.0], float)
        c1 = 0.10 - m
        c2 = 1e-3 - sse
        c3 = 1e-3 + sse
        return np.array([c1, c2, c3], float)

    bnds = Bounds(lb=[-10.0, -10.0, -10.0], ub=[10.0, 10.0, 10.0])

    def consFun(k):
        return consVals(k)

    slsqpCons = [
        {'type': 'ineq', 'fun': lambda k: consFun(k)[0]},
        {'type': 'ineq', 'fun': lambda k: consFun(k)[1]},
        {'type': 'ineq', 'fun': lambda k: consFun(k)[2]},
    ]

    k0 = np.asarray(initK, float)
    initData = once(k0, count=False)
    initT, initY, initR, initTs, initM, initSse, initOk = initData
    initFeas = (initM <= 0.10) and (abs(initSse) <= 1e-3)
    initFeasRes = feasibilityResidual(k0, initM, initSse)
    initElapsed = time.time() - startTime
    initGradNorm = gradNormEst(k0, f0=initTs)
    logRow(0, k0, initTs, initM, initSse, initFeas, initFeasRes, initElapsed, initGradNorm)
    savePlot(k0, idx=0, data=initData)
    lastLogged['k'] = np.asarray(k0, float)
    print('start solve (SLSQP)')

    def callback(k, state=None):
        if lastLogged['k'] is not None and np.allclose(k, lastLogged['k'], rtol=1e-12, atol=1e-12):
            return
        t, y, r, ts, m, sse, ok = once(k, count=False)
        feas = (m <= 0.10) and (abs(sse) <= 1e-3)
        elapsed = time.time() - startTime
        gn = gradNormEst(k, f0=ts)
        feasRes = feasibilityResidual(k, m, sse)
        iterIdx = lastLogged['idx'] + 1
        print(f'SLSQPiter {iterIdx:03d}: kp {k[0]:.5g} ki {k[1]:.5g} kd {k[2]:.5g}  Ts {ts:.5g}  M {m:.5g}  SSe {sse:.3e}  feas {feas}  feasRes {feasRes:.3g}  Time(s) {elapsed:.2f}  gradnorm {gn:.3g}  funCalls {simCount["n"]}')
        logRow(iterIdx, k, ts, m, sse, feas, feasRes, elapsed, gn)
        savePlot(k, idx=iterIdx, data=(t, y, r, ts, m, sse, ok))

    res = minimize(fun, k0,
                   method='SLSQP',
                   jac=None,
                   constraints=slsqpCons,
                   bounds=bnds,
                   callback=callback,
                   options={'maxiter': 200, 'ftol': 1e-5, 'disp': True, 'eps': 1.4901161193847656e-08})

    print('solve done')
    print('best gains')
    print('kp', res.x[0], 'ki', res.x[1], 'kd', res.x[2])

    # Ensure final iterate is logged/plotted if not identical to last callback
    if lastLogged['k'] is None or not np.allclose(res.x, lastLogged['k'], rtol=1e-12, atol=1e-12):
        kFinal = np.asarray(res.x, float)
        t, y, r, ts, m, sse, ok = once(kFinal, count=False)
        feas = (m <= 0.10) and (abs(sse) <= 1e-3)
        elapsed = time.time() - startTime
        gn = gradNormEst(kFinal, f0=ts)
        feasRes = feasibilityResidual(kFinal, m, sse)
        iterIdx = lastLogged['idx'] + 1
        logRow(iterIdx, kFinal, ts, m, sse, feas, feasRes, elapsed, gn)
        savePlot(kFinal, idx=iterIdx, data=(t, y, r, ts, m, sse, ok))

    print('\nSLSQP details:')
    print(' status:', getattr(res, 'status', None))
    print(' message:', getattr(res, 'message', None))
    print(' niter:', getattr(res, 'nit', None))
    print(' nfev:', getattr(res, 'nfev', None))

    return res


if __name__ == "__main__":
    runOptimization(np.array([0.5, 0.5, 0.5]))
