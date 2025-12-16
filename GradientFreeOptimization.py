import os
import time
import csv
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matlab.engine  # type: ignore
from scipy.optimize import minimize, Bounds, differential_evolution  # type: ignore

# Toggles
RUN_NM = True
RUN_DE = False      # Differential Evolution (serial)

# MATLAB setup (shared engine)
eng = matlab.engine.connect_matlab('engine_1')
model = 'ProofofConceptProblem'
eng.load_system(model, nargout=0)
eng.set_param(model, 'SimulationCommand', 'stop', nargout=0)
eng.set_param(model, 'SimulationMode', 'accelerator', nargout=0)  # speed
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


def runOptimization(initK, baseDir=None, alg="NM"):
    label = buildLabel(initK)
    outDir = baseDir if baseDir else f"slicesGF_{label}"
    os.makedirs(outDir, exist_ok=True)
    logPath = os.path.join(outDir, f"opt_log_{label}.csv")

    startTime = time.time()
    simCount = {'n': 0}
    evalCount = {'n': 0}
    lastLogged = {'idx': 0, 'k': None}

    with open(logPath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'kp', 'ki', 'kd', 'Ts', 'M', 'SSe', 'feas', 'feasResidual', 'Time(s)', 'gradnorm', 'funCalls'])

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
            if r is not None and r.size == y.size:
                plt.plot(t, r, '--', label='r')
            elif r is not None:
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

    def penalized_cost(k, tag="Eval"):
        _, _, _, ts, m, sse, ok = once(k, count=True)
        evalCount['n'] += 1
        feas = ok and np.isfinite(ts) and np.isfinite(m) and np.isfinite(sse)
        viol = max(0.0, m - 0.10)
        viol = max(viol, abs(sse) - 1e-3)
        for ki, bound in zip(k, (-10.0, -10.0, -10.0)):
            viol = max(viol, bound - ki)
        for ki, bound in zip(k, (10.0, 10.0, 10.0)):
            viol = max(viol, ki - bound)
        penalty = 1e3 * viol * viol
        cost = ts if (feas and np.isfinite(ts)) else 1e6
        total = cost + penalty
        print(f'{tag}{evalCount["n"]:04d}: kp {k[0]:.16g} ki {k[1]:.16g} kd {k[2]:.16g}  Ts {ts:.6g}  M {m:.6g}  SSe {sse:.6g}  viol {viol:.3g}  cost {total:.6g}')
        return total

    bnds = Bounds(lb=[-10.0, -10.0, -10.0], ub=[10.0, 10.0, 10.0])

    # log initial point once for NM/DE
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

    if alg == "NM":
        print('start solve (Nelder-Mead)')
        nm_iter = {'i': 0}

        def callback(xk):
            nm_iter['i'] += 1
            t, y, r, ts, m, sse, ok = once(xk, count=False)
            feas = (m <= 0.10) and (abs(sse) <= 1e-3)
            elapsed = time.time() - startTime
            gn = gradNormEst(xk, f0=ts)
            feasRes = feasibilityResidual(xk, m, sse)
            iterIdx = nm_iter['i']
            print(f'NMiter {iterIdx:03d}: kp {xk[0]:.5g} ki {xk[1]:.5g} kd {xk[2]:.5g}  Ts {ts:.5g}  M {m:.5g}  SSe {sse:.3e}  feas {feas}  feasRes {feasRes:.3g}  Time(s) {elapsed:.2f}  gradnorm {gn:.3g}  funCalls {simCount["n"]}')
            logRow(iterIdx, xk, ts, m, sse, feas, feasRes, elapsed, gn)
            savePlot(xk, idx=iterIdx, data=(t, y, r, ts, m, sse, ok))

        res = minimize(
            lambda k: penalized_cost(k, tag="NMeval "),
            k0,
            method='Nelder-Mead',
            callback=callback,
            options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': True}
        )

    elif alg == "DE":
        print('start solve (Differential Evolution)')
        de_iter = {'i': 0}

        def de_callback(xk, convergence=None):
            de_iter['i'] += 1
            t, y, r, ts, m, sse, ok = once(xk, count=False)
            feas = (m <= 0.10) and (abs(sse) <= 1e-3)
            elapsed = time.time() - startTime
            gn = gradNormEst(xk, f0=ts)
            feasRes = feasibilityResidual(xk, m, sse)
            iterIdx = de_iter['i']
            print(f'DEiter {iterIdx:03d}: kp {xk[0]:.5g} ki {xk[1]:.5g} kd {xk[2]:.5g}  Ts {ts:.5g}  M {m:.5g}  SSe {sse:.3e}  feas {feas}  feasRes {feasRes:.3g}  Time(s) {elapsed:.2f}  gradnorm {gn:.3g}  funCalls {simCount["n"]}')
            logRow(iterIdx, xk, ts, m, sse, feas, feasRes, elapsed, gn)
            savePlot(xk, idx=iterIdx, data=(t, y, r, ts, m, sse, ok))
            return False

        res = differential_evolution(
            lambda k: penalized_cost(k, tag="DEeval "),
            bounds=[(-10.0, 10.0)] * 3,
            maxiter=50,
            popsize=15,
            polish=False,
            callback=de_callback,
            disp=True,
            tol=0.01,
            workers=None,
        )

    else:
        raise ValueError("Unknown algorithm")

    print('solve done')
    print('best gains')
    print('kp', res.x[0], 'ki', res.x[1], 'kd', res.x[2])

    # final log/plot
    kFinal = np.asarray(res.x, float)
    t, y, r, ts, m, sse, ok = once(kFinal, count=False)
    feas = (m <= 0.10) and (abs(sse) <= 1e-3)
    elapsed = time.time() - startTime
    gn = gradNormEst(kFinal, f0=ts)
    feasRes = feasibilityResidual(kFinal, m, sse)
    iterIdx = lastLogged['idx'] + 1
    logRow(iterIdx, kFinal, ts, m, sse, feas, feasRes, elapsed, gn)
    savePlot(kFinal, idx=iterIdx, data=(t, y, r, ts, m, sse, ok))

    print(f'\n{alg} details:')
    print(' status:', getattr(res, 'message', None))
    print(' niter:', getattr(res, 'nit', None))
    print(' nfev:', getattr(res, 'nfev', None))

    return res


if __name__ == "__main__":
    if RUN_NM:
        runOptimization(np.array([0.5, 0.5, 0.5]), alg="NM")
    if RUN_DE:
        runOptimization(np.array([0.5, 0.5, 0.5]), alg="DE")
