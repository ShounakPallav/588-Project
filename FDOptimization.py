import os
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from scipy.optimize import minimize, Bounds, NonlinearConstraint

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
    while j > 0 and err[j-1] <= band:
        j -= 1
    if j == 0:
        return float(t[i0])
    # linear interpolate the crossing near j-1 -> j
    e1 = err[j-1] - band
    e2 = err[j]   - band
    if e1 == e2:
        return float(t[j])
    alpha = -e1 / (e2 - e1)
    tcross = t[j-1] + alpha * (t[j] - t[j-1])
    return float(tcross)

eng = matlab.engine.connect_matlab('engine_1')
model = 'ProofofConceptProblem'
eng.load_system(model, nargout=0)
# make sure nothing is running
eng.set_param(model, 'SimulationCommand', 'stop', nargout=0)
eng.set_param(model, 'SolverType', 'Variable-step', nargout=0)
eng.set_param(model, 'Solver', 'ode45', nargout=0)
eng.set_param(model, 'RelTol', '1e-6', nargout=0)
eng.set_param(model, 'AbsTol', '1e-8', nargout=0)
eng.set_param(model, 'MaxStep', '1e-3', nargout=0)

if not os.path.isdir('slices'):
    os.makedirs('slices')

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
        print(f'eval {evalcount["n"]:04d}: kp {k[0]:.4g} ki {k[1]:.4g} kd {k[2]:.4g}  unstable or bad -> large cost')
        return 1e6
    print(f'eval {evalcount["n"]:04d}: kp {k[0]:.4g} ki {k[1]:.4g} kd {k[2]:.4g}  Ts {ts:.4g}  M {m:.4g}  SSe {sse:.2e}')
    return ts

def grad(k):
    k = np.asarray(k, float)
    # relative step ~5% per component, with a floor so zeros move
    h = np.maximum(0.05*np.maximum(1.0, np.abs(k)), 0.02)
    g = np.zeros_like(k)
    f0 = fun(k)
    for i in range(len(k)):
        e = np.zeros_like(k); e[i] = 1.0
        hp = h[i]
        fp = fun(k + hp*e)
        fm = fun(k - hp*e)
        if np.isfinite(fp) and np.isfinite(fm):
            g[i] = (fp - fm) / (2*hp)
        elif np.isfinite(fp):
            g[i] = (fp - f0) / hp
        elif np.isfinite(fm):
            g[i] = (f0 - fm) / hp
        else:
            g[i] = 0.0
    print(f'grad norm {np.linalg.norm(g):.4g}')
    return g

def consvals(k):
    _, _, _, ts, m, sse, ok = once(k)
    if not ok or not np.isfinite(m) or not np.isfinite(sse):
        return np.array([-1.0, -1.0, -1.0], float)
    c1 = 0.10 - m      # M ≤ 0.10
    c2 = 1e-3 - sse    #  sse ≤ 1e-3
    c3 = 1e-3 + sse    # -sse ≤ 1e-3
    return np.array([c1, c2, c3], float)

def consjac(k, h=1e-2):
    base = consvals(k)
    J = np.zeros((3, len(k)), float)
    for i in range(len(k)):
        e = np.zeros_like(k); e[i] = 1.0
        step = h
        col = None
        for _ in range(3):
            vp = consvals(k + step*e)
            vm = consvals(k - step*e)
            if np.all(np.isfinite(vp)) and np.all(np.isfinite(vm)):
                col = (vp - vm) / (2*step); break
            if np.all(np.isfinite(vp)):
                col = (vp - base) / step; break
            if np.all(np.isfinite(vm)):
                col = (base - vm) / step; break
            step *= 0.1
        J[:, i] = np.zeros(3) if col is None else col
    return J

it = {'n': 0}

def saveplot(k):
    t, y, r, ts, m, sse, ok = once(k)
    plt.figure()
    if t is not None and y is not None:
        plt.plot(t, y, label='y')
        if r.size == y.size:
            plt.plot(t, r, '--', label='r')
        else:
            val = float(r) if r.size == 1 else tailvalue(r)
            plt.axhline(val, linestyle='--', label='r')
    ttl = f'kp={k[0]:.3g} ki={k[1]:.3g} kd={k[2]:.3g}  Ts={ts:.3g}  M={m:.3g}  SSe={sse:.2e}'
    plt.title(ttl)
    plt.xlabel('Time'); plt.ylabel('Value'); plt.grid(True); plt.legend(); plt.tight_layout()
    name = f'slices/iter{it["n"]:03d}.pdf'
    plt.savefig(name)
    plt.close()

def callback(k, state=None):
    t, y, r, ts, m, sse, ok = once(k)
    feas = (m <= 0.10) and (abs(sse) <= 1e-3)
    print(f'iter {it["n"]:03d}: kp {k[0]:.5g} ki {k[1]:.5g} kd {k[2]:.5g}  Ts {ts:.5g}  M {m:.5g}  SSe {sse:.3e}  feas {feas}')
    saveplot(k)
    it['n'] += 1

bnds = Bounds(lb=[-200.0, -200.0, -200.0], ub=[200.0, 200.0, 200.0])

def cons_fun(k):
    return consvals(k)

def cons_jac_fun(k):
    return consjac(k)

nlc = NonlinearConstraint(fun=cons_fun,
                          lb=[0.0, 0.0, 0.0],
                          ub=[np.inf, np.inf, np.inf],
                          jac=cons_jac_fun)

x0 = np.array([5.0, 5.0, 5.0])

print('start solve (trust-constr, verbose=3)')
res = minimize(fun, x0,
               method='trust-constr',
               jac=grad,                    # ← change this line
               constraints=[nlc],
               bounds=bnds,
               callback=callback,
               options={'maxiter': 200, 'verbose': 3, 'gtol': 1e-6, 'xtol': 1e-8, 'barrier_tol': 1e-8})

print('solve done')

print('best gains')
print('kp', res.x[0], 'ki', res.x[1], 'kd', res.x[2])

saveplot(res.x)
