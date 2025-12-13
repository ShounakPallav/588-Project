import os
import time
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matlab.engine  # type: ignore
from scipy.optimize import minimize, Bounds  # type: ignore

# Connect to shared MATLAB session
eng = matlab.engine.connect_matlab('engine_1')
model = 'ProofofConceptProblem'
eng.load_system(model, nargout=0)
# Make sure nothing is running and set solver options
eng.set_param(model, 'SimulationCommand', 'stop', nargout=0)
eng.set_param(model, 'FastRestart', 'off', nargout=0)
eng.set_param(model, 'SolverType', 'Variable-step', nargout=0)
eng.set_param(model, 'Solver', 'ode45', nargout=0)
eng.set_param(model, 'StopTime', '25', nargout=0)
eng.set_param(model, 'RelTol', '1e-6', nargout=0)
eng.set_param(model, 'AbsTol', '1e-8', nargout=0)
eng.set_param(model, 'MaxStep', '1e-3', nargout=0)

if not os.path.isdir('slices'):
    os.makedirs('slices')


def tovec(x):
    return np.asarray(x).squeeze()


def tailvalue(y, n=20):
    n = min(n, y.size)
    return float(np.mean(y[-n:]))


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
    # walk back to the last point just before staying in band
    j = i0
    while j > 0 and err[j - 1] <= band:
        j -= 1
    if j == 0:
        return float(t[i0])
    # linear interpolate the crossing near j-1 -> j
    e1 = err[j - 1] - band
    e2 = err[j] - band
    if e1 == e2:
        return float(t[j])
    alpha = -e1 / (e2 - e1)
    tcross = t[j - 1] + alpha * (t[j] - t[j - 1])
    return float(tcross)


def saferun(k, ymax=1e3):
    try:
        kp, ki, kd = float(k[0]), float(k[1]), float(k[2])
        eng.workspace['kp'] = kp
        eng.workspace['ki'] = ki
        eng.workspace['kd'] = kd
        eng.sim(model, nargout=0)
        t = tovec(eng.workspace['time'])
        y = tovec(eng.workspace['y'])
        r = tovec(eng.workspace['r'])
    except Exception:
        return None, None, None, np.inf, np.inf, np.inf, False

    if t.size == 0 or y.size == 0 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        return t, y, r, np.inf, np.inf, np.inf, False
    if np.max(np.abs(y)) > ymax:
        return t, y, r, np.inf, np.inf, np.inf, False

    yss = tailvalue(y, max(10, y.size // 10))
    rss = tailvalue(r, max(10, r.size // 10)) if r.size > 1 else float(r)
    m = (np.max(y) - yss) / (yss if yss != 0 else 1.0)
    ts = settle(t, y, yss, pct=0.02)
    sse = rss - yss

    if not np.isfinite(ts):
        return t, y, r, np.inf, max(m, 0.11), np.inf, False

    return t, y, r, ts, m, sse, True


def once(k):
    t, y, r, ts, m, sse, ok = saferun(k)
    return t, y, r, ts, m, sse, ok


evalcount = {'n': 0}


def fun(k):
    _, _, _, ts, m, sse, ok = once(k)
    evalcount['n'] += 1
    if not ok:
        print(f'FDeval {evalcount["n"]:04d}: kp {k[0]:.4g} ki {k[1]:.4g} kd {k[2]:.4g}  unstable or bad -> large cost')
        return 1e6
    print(f'FDeval {evalcount["n"]:04d}: kp {k[0]:.4g} ki {k[1]:.4g} kd {k[2]:.4g}  Ts {ts:.4g}  M {m:.4g}  SSe {sse:.2e}')
    return ts


def consvals(k):
    _, _, _, ts, m, sse, ok = once(k)
    if not ok or not np.isfinite(m) or not np.isfinite(sse):
        return np.array([-1.0, -1.0, -1.0], float)
    c1 = 0.10 - m      # overshoot <= 0.10
    c2 = 1e-3 - sse    # sse <= 1e-3
    c3 = 1e-3 + sse    # -sse <= 1e-3  (i.e., sse >= -1e-3)
    return np.array([c1, c2, c3], float)


it = {'n': 0}


def saveplot(k, idx=None):
    t, y, r, ts, m, sse, ok = once(k)
    plt.figure()
    ss_val = None
    if t is not None and y is not None:
        ss_val = tailvalue(y, max(10, y.size // 10))
        plt.plot(t, y, label='y')
        if r.size == y.size:
            plt.plot(t, r, '--', label='r')
        else:
            val = float(r) if r.size == 1 else tailvalue(r)
            plt.axhline(val, linestyle='--', label='r')
        if ss_val is not None:
            plt.axhline(0.98 * ss_val, color='gray', linestyle='--', linewidth=1.0, label='0.98·y_ss')
            plt.axhline(1.02 * ss_val, color='gray', linestyle='--', linewidth=1.0, label='1.02·y_ss')
            plt.axhline(1.10 * ss_val, color='red', linestyle='--', linewidth=1.0, label='OS 10% bound')
    ttl = f'kp={k[0]:.3g} ki={k[1]:.3g} kd={k[2]:.3g}  Ts={ts:.3g}  M={m:.3g}  SSe={sse:.2e}'
    plt.title(ttl)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_idx = it['n'] if idx is None else idx
    name = f'slices/iter{plot_idx:03d}.pdf'
    plt.savefig(name)
    plt.close()


def callback(k, state=None):
    t, y, r, ts, m, sse, ok = once(k)
    feas = (m <= 0.10) and (abs(sse) <= 1e-3)
    elapsed = time.time() - t0
    print(f'SLSQPiter {it["n"]:03d}: kp {k[0]:.5g} ki {k[1]:.5g} kd {k[2]:.5g}  Ts {ts:.5g}  M {m:.5g}  SSe {sse:.3e}  feas {feas}  Time(s) {elapsed:.2f}')
    saveplot(k, idx=it['n'])
    it['n'] += 1


bnds = Bounds(lb=[-200.0, -200.0, -200.0], ub=[200.0, 200.0, 200.0])


def cons_fun(k):
    return consvals(k)


# SLSQP expects constraint functions >= 0; use scalar constraints and let it finite-difference.
slsqp_cons = [
    {'type': 'ineq', 'fun': lambda k: cons_fun(k)[0]},
    {'type': 'ineq', 'fun': lambda k: cons_fun(k)[1]},
    {'type': 'ineq', 'fun': lambda k: cons_fun(k)[2]},
]


x0 = np.array([5.0, 5.0, 5.0])

saveplot(x0, idx=0)
it['n'] = 1
print('start solve (SLSQP)')

t0 = time.time()

res = minimize(fun, x0,
               method='SLSQP',
               jac=None,                    # let SLSQP finite-difference
               constraints=slsqp_cons,
               bounds=bnds,
               callback=callback,
               options={'maxiter': 200, 'ftol': 1e-5, 'disp': True})

print('solve done')

print('best gains')
print('kp', res.x[0], 'ki', res.x[1], 'kd', res.x[2])

saveplot(res.x)

print('\nSLSQP details:')
print(' status:', getattr(res, 'status', None))
print(' message:', getattr(res, 'message', None))
print(' niter:', getattr(res, 'nit', None))
print(' nfev:', getattr(res, 'nfev', None))
