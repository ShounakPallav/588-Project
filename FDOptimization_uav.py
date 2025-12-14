import os
import time
import csv
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matlab.engine  # type: ignore
from scipy.optimize import minimize, Bounds  # type: ignore

# ------------------------------------------------------------
# MATLAB setup (shared engine) + model configuration
# ------------------------------------------------------------
eng = matlab.engine.connect_matlab('engine_1')
model = 'uav_problem'  # Simulink model name

eng.load_system(model, nargout=0)
eng.set_param(model, 'SimulationCommand', 'stop', nargout=0)
eng.set_param(model, 'FastRestart', 'off', nargout=0)
eng.set_param(model, 'SolverType', 'Variable-step', nargout=0)
eng.set_param(model, 'Solver', 'ode45', nargout=0)
eng.set_param(model, 'StopTime', '30', nargout=0)
eng.set_param(model, 'RelTol', '1e-6', nargout=0)
eng.set_param(model, 'AbsTol', '1e-8', nargout=0)
eng.set_param(model, 'MaxStep', '1e-3', nargout=0)

# Design vector order:
# k[0] = Kp_h, k[1] = Ki_h,
# k[2] = Kp_th, k[3] = Ki_th, k[4] = Kd_th, k[5] = N_th,
# k[6] = Kp_V, k[7] = Ki_V,
# k[8] = Kaw,  k[9] = Kff_gam
DESIGN_NAMES = ["Kp_h", "Ki_h", "Kp_th", "Ki_th", "Kd_th",
                "N_th", "Kp_V", "Ki_V", "Kaw", "Kff_gam"]


def init_uav_params_in_workspace(eng):
    """
    Push all initial UAV parameters into the MATLAB base workspace
    using eng.workspace[...] and build struct p in MATLAB.
    Mirrors uav_params_init.m (without clear/close).
    """

    # --- Vehicle / environment ---
    m    = 13.5
    S    = 0.55
    c    = 0.30
    Iy   = 1.2
    Tmax = 50.0
    g    = 9.81

    rho0 = 1.225
    H    = 8500.0

    # --- Aero coefficients ---
    CL0  = 0.30
    CLa  = 5.50
    CLq  = 7.50
    CLde = 0.35

    CD0  = 0.03
    k_ind = 0.06  # avoid shadowing Python 'k'

    CM0  = 0.02
    CMa  = -1.0
    CMq  = -12.5
    CMde = -1.10

    # push scalars into MATLAB workspace
    eng.workspace['m']    = m
    eng.workspace['S']    = S
    eng.workspace['c']    = c
    eng.workspace['Iy']   = Iy
    eng.workspace['Tmax'] = Tmax
    eng.workspace['g']    = g

    eng.workspace['rho0'] = rho0
    eng.workspace['H']    = H

    eng.workspace['CL0']  = CL0
    eng.workspace['CLa']  = CLa
    eng.workspace['CLq']  = CLq
    eng.workspace['CLde'] = CLde

    eng.workspace['CD0']  = CD0
    eng.workspace['k_ind'] = k_ind

    eng.workspace['CM0']  = CM0
    eng.workspace['CMa']  = CMa
    eng.workspace['CMq']  = CMq
    eng.workspace['CMde'] = CMde

    # build struct p in MATLAB
    eng.eval("""
        p = struct();
        p.m    = m;
        p.S    = S;
        p.c    = c;
        p.Iy   = Iy;
        p.Tmax = Tmax;
        p.g    = g;

        p.rho0 = rho0;
        p.H    = H;

        p.CL0  = CL0;
        p.CLa  = CLa;
        p.CLq  = CLq;
        p.CLde = CLde;

        p.CD0  = CD0;
        p.k    = k_ind;

        p.CM0  = CM0;
        p.CMa  = CMa;
        p.CMq  = CMq;
        p.CMde = CMde;
    """, nargout=0)

    # --- Actuator limits ---
    de_max_deg = 25.0
    de_min_deg = -25.0
    de_max = float(np.deg2rad(de_max_deg))
    de_min = float(np.deg2rad(de_min_deg))

    dt_min = 0.0
    dt_max = 1.0

    eng.workspace['de_max_deg'] = de_max_deg
    eng.workspace['de_min_deg'] = de_min_deg
    eng.workspace['de_max']     = de_max
    eng.workspace['de_min']     = de_min
    eng.workspace['dt_min']     = dt_min
    eng.workspace['dt_max']     = dt_max

    # --- Initial conditions ("trim-ish") ---
    h0   = 100.0
    V0   = 20.0
    gam0 = 0.0
    q0   = 0.0

    rho  = rho0 * np.exp(-h0 / H)
    qbar = 0.5 * rho * V0**2
    CLreq  = (m * g) / (qbar * S)
    alpha0 = (CLreq - CL0) / CLa
    th0    = alpha0 + gam0

    eng.workspace['h0']   = h0
    eng.workspace['V0']   = V0
    eng.workspace['gam0'] = gam0
    eng.workspace['q0']   = q0
    eng.workspace['alpha0'] = alpha0
    eng.workspace['th0']    = th0

    # --- Command profile settings ---
    dh          = 10.0
    h_step_time = 1.0
    Vcmd0       = V0

    eng.workspace['dh']          = dh
    eng.workspace['h_step_time'] = h_step_time
    eng.workspace['Vcmd0']       = Vcmd0

    # --- Initial controller gains (just to start; optimizer overwrites these) ---
    Kp_h    = 0.04
    Ki_h    = 0.01

    Kp_th   = 2.0
    Ki_th   = 0.6
    Kd_th   = 0.25
    N_th    = 20.0

    Kp_V    = 0.3
    Ki_V    = 0.08

    Kaw     = 5.0
    Kff_gam = 0.0

    eng.workspace['Kp_h']   = Kp_h
    eng.workspace['Ki_h']   = Ki_h

    eng.workspace['Kp_th']  = Kp_th
    eng.workspace['Ki_th']  = Ki_th
    eng.workspace['Kd_th']  = Kd_th
    eng.workspace['N_th']   = N_th

    eng.workspace['Kp_V']   = Kp_V
    eng.workspace['Ki_V']   = Ki_V

    eng.workspace['Kaw']    = Kaw
    eng.workspace['Kff_gam'] = Kff_gam

    # --- Simulation settings ---
    tstop = 30.0
    eng.workspace['tstop'] = tstop


# initialize parameters in MATLAB
init_uav_params_in_workspace(eng)


def fmtVal(val):
    return f"{val:.6g}".replace('+', '').replace(' ', '')


def buildLabel(k):
    # just use first three for a short label
    return f"Kph{fmtVal(k[0])}_Kih{fmtVal(k[1])}_Kpth{fmtVal(k[2])}"


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


def step_metrics(t, y, r, pct_band=0.02):
    """
    Compute settling time, overshoot, steady-state error for a step.
    y: output, r: reference (scalar or vector).
    """
    if t.size == 0 or y.size == 0:
        return np.inf, np.inf, np.inf

    yss = tailValue(y, max(10, y.size // 10))
    if np.size(r) > 1:
        rss = tailValue(toVec(r), max(10, np.size(r) // 10))
    else:
        rss = float(r)

    M = (np.max(y) - yss) / (yss if yss != 0 else 1.0)
    Ts = settle(t, y, yss, pct=pct_band)
    SSe = rss - yss
    return Ts, M, SSe


def feasibilityResidual(M_h, SSe_h, M_V, SSe_V, Ts_V):
    """
    Max violation of constraints:
    M_h <= 0.40
    |SSe_h| <= 1e-3
    Ts_V <= 20
    M_V <= 0.40
    |SSe_V| <= 1e-3
    """
    viol = 0.0
    viol = max(viol, M_h - 0.40)
    viol = max(viol, abs(SSe_h) - 1e-3)
    viol = max(viol, Ts_V - 20.0)
    viol = max(viol, M_V - 0.40)
    viol = max(viol, abs(SSe_V) - 1e-3)
    return viol


def runOptimization(initK, baseDir=None):
    k0 = np.asarray(initK, float)
    label = buildLabel(k0)
    outDir = baseDir if baseDir else f"slices_{label}"
    os.makedirs(outDir, exist_ok=True)
    logPath = os.path.join(outDir, f"opt_log_{label}.csv")

    startTime = time.time()
    simCount = {'n': 0}
    evalCount = {'n': 0}
    lastLogged = {'idx': 0, 'k': None}

    # CSV header
    with open(logPath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['SLSQPiter'] +
            DESIGN_NAMES +
            ['Ts_h', 'M_h', 'SSe_h', 'Ts_V', 'M_V', 'SSe_V',
             'feas', 'feasResidual', 'Time(s)', 'gradnorm', 'funCalls']
        )

    def saferun(k, ymax=1e3, count=False):
        """
        Run Simulink for given design vector k and compute metrics
        for height and velocity.
        """
        try:
            k = np.asarray(k, float)

            # unpack design vars to MATLAB workspace
            eng.workspace['Kp_h']    = float(k[0])
            eng.workspace['Ki_h']    = float(k[1])
            eng.workspace['Kp_th']   = float(k[2])
            eng.workspace['Ki_th']   = float(k[3])
            eng.workspace['Kd_th']   = float(k[4])
            eng.workspace['N_th']    = float(k[5])
            eng.workspace['Kp_V']    = float(k[6])
            eng.workspace['Ki_V']    = float(k[7])
            eng.workspace['Kaw']     = float(k[8])
            eng.workspace['Kff_gam'] = float(k[9])

            # run Simulink model
            eng.sim(model, nargout=0)
            if count:
                simCount['n'] += 1

            # pull signals from workspace
            t      = toVec(eng.workspace['time'])
            h_ref  = toVec(eng.workspace['h_ref'])
            V_ref  = toVec(eng.workspace['V_ref'])
            h_out  = toVec(eng.workspace['h_f_out'])
            V_out  = toVec(eng.workspace['v_f_out'])
        except Exception:
            return (None, None, None, None, None,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, False)

        # sanity checks
        if (t.size == 0 or h_out.size == 0 or V_out.size == 0 or
                np.any(~np.isfinite(t)) or np.any(~np.isfinite(h_out)) or np.any(~np.isfinite(V_out))):
            return (t, h_out, V_out, h_ref, V_ref,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, False)

        if np.max(np.abs(h_out)) > ymax or np.max(np.abs(V_out)) > ymax:
            return (t, h_out, V_out, h_ref, V_ref,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, False)

        # metrics
        Ts_h, M_h, SSe_h = step_metrics(t, h_out, h_ref)
        Ts_V, M_V, SSe_V = step_metrics(t, V_out, V_ref)

        if not np.isfinite(Ts_h):
            return (t, h_out, V_out, h_ref, V_ref,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, False)

        return (t, h_out, V_out, h_ref, V_ref,
                Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, True)

    def once(k, count=False):
        return saferun(k, count=count)

    def evalCostSilent(k):
        *_, Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, ok = once(k, count=False)
        if not ok or not np.isfinite(Ts_h):
            return np.inf
        return Ts_h  # objective = height settling time

    def gradNormEst(k, f0=None, h0=1e-2):
        k = np.asarray(k, float)
        g = np.zeros_like(k)
        fBase = evalCostSilent(k) if f0 is None else f0
        for i in range(len(k)):
            step = max(h0, 0.05 * max(1.0, abs(k[i])))
            e = np.zeros_like(k)
            e[i] = 1.0
            kp = k + step * e
            km = k - step * e
            fp = evalCostSilent(kp)
            fm = evalCostSilent(km)
            if np.isfinite(fp) and np.isfinite(fm):
                g[i] = (fp - fm) / (2 * step)
            elif np.isfinite(fp):
                g[i] = (fp - fBase) / step
            elif np.isfinite(fm):
                g[i] = (fBase - fm) / step
            else:
                g[i] = 0.0
        return float(np.linalg.norm(g))

    def logRow(iterIdx, k, Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V,
               feas, feasRes, elapsed, gn):
        with open(logPath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [iterIdx] +
                [float(val) for val in k] +
                [Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V,
                 feas, feasRes, elapsed, gn, simCount['n']]
            )
        lastLogged['idx'] = iterIdx
        lastLogged['k'] = np.asarray(k, float)

    def savePlot(k, idx=None, data=None):
        if data is None:
            t, h_out, V_out, h_ref, V_ref, Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, ok = once(k, count=False)
        else:
            t, h_out, V_out, h_ref, V_ref, Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, ok = data

        plt.figure()
        if t is not None and h_out is not None:
            plt.plot(t, h_out, label='h')
            plt.plot(t, h_ref, '--', label='h_ref')
        if t is not None and V_out is not None:
            plt.plot(t, V_out, label='V')
            plt.plot(t, V_ref, '--', label='V_ref')

        ttl = (f'Ts_h={Ts_h:.3g}, M_h={M_h:.3g}, SSe_h={SSe_h:.2e}  '
               f'Ts_V={Ts_V:.3g}, M_V={M_V:.3g}, SSe_V={SSe_V:.2e}')
        plt.title(ttl)
        plt.xlabel('Time')
        plt.ylabel('States')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plotIdx = lastLogged['idx'] if idx is None else idx
        name = os.path.join(outDir, f'iter{plotIdx:03d}_{label}.pdf')
        plt.savefig(name)
        plt.close()

    def fun(k):
        t, h_out, V_out, h_ref, V_ref, Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, ok = once(k, count=True)
        evalCount['n'] += 1
        if not ok:
            print(f'FDeval {evalCount["n"]:04d}: k={k}  unstable/bad -> large cost')
            return 1e6
        print(f'FDeval {evalCount["n"]:04d}: k={k}  Ts_h {Ts_h:.6g}  M_h {M_h:.6g}  '
              f'SSe_h {SSe_h:.3e}  Ts_V {Ts_V:.6g}  M_V {M_V:.6g}  SSe_V {SSe_V:.3e}')
        return Ts_h  # objective

    def consVals(k):
        _, _, _, _, _, Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, ok = once(k, count=True)
        if not ok or not np.isfinite(M_h) or not np.isfinite(SSe_h):
            return np.array([-1.0] * 7, float)

        # c >= 0 style inequalities
        c1 = 0.40 - M_h         # height overshoot <= 40%
        c2 = 1e-3 - SSe_h       # SSe_h <= 1e-3
        c3 = 1e-3 + SSe_h       # -SSe_h <= 1e-3
        c4 = 20.0 - Ts_V        # Ts_V <= 20
        c5 = 0.40 - M_V         # velocity overshoot <= 40%
        c6 = 1e-3 - SSe_V       # SSe_V <= 1e-3
        c7 = 1e-3 + SSe_V       # -SSe_V <= 1e-3
        return np.array([c1, c2, c3, c4, c5, c6, c7], float)

    # Bounds: simple -10..10 on each design variable
    bnds = Bounds(lb=[-10.0] * len(k0), ub=[10.0] * len(k0))

    def consFun(k):
        return consVals(k)

    slsqpCons = [
        {'type': 'ineq', 'fun': (lambda k, idx=i: consFun(k)[idx])}
        for i in range(7)
    ]

    # Initial iterate metrics / log
    initData = once(k0, count=False)
    (initT, initH, initV, initHref, initVref,
     initTs_h, initM_h, initSSe_h, initTs_V, initM_V, initSSe_V, initOk) = initData

    initFeas = (initM_h <= 0.40 and abs(initSSe_h) <= 1e-3 and
                initTs_V <= 20.0 and initM_V <= 0.40 and abs(initSSe_V) <= 1e-3)
    initFeasRes = feasibilityResidual(initM_h, initSSe_h, initM_V, initSSe_V, initTs_V)
    initElapsed = time.time() - startTime
    initGradNorm = gradNormEst(k0, f0=initTs_h)
    logRow(0, k0, initTs_h, initM_h, initSSe_h,
           initTs_V, initM_V, initSSe_V, initFeas, initFeasRes,
           initElapsed, initGradNorm)
    savePlot(k0, idx=0, data=initData)
    lastLogged['k'] = np.asarray(k0, float)
    print('start solve (SLSQP)')

    def callback(k, state=None):
        if lastLogged['k'] is not None and np.allclose(k, lastLogged['k'],
                                                      rtol=1e-12, atol=1e-12):
            return
        data = once(k, count=False)
        (_, _, _, _, _,
         Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, ok) = data
        feas = (M_h <= 0.40 and abs(SSe_h) <= 1e-3 and
                Ts_V <= 20.0 and M_V <= 0.40 and abs(SSe_V) <= 1e-3)
        elapsed = time.time() - startTime
        gn = gradNormEst(k, f0=Ts_h)
        feasRes = feasibilityResidual(M_h, SSe_h, M_V, SSe_V, Ts_V)
        iterIdx = lastLogged['idx'] + 1
        print(f'SLSQPiter {iterIdx:03d}: k={k}  Ts_h {Ts_h:.5g}  M_h {M_h:.5g}  '
              f'SSe_h {SSe_h:.3e}  Ts_V {Ts_V:.5g}  M_V {M_V:.5g}  '
              f'SSe_V {SSe_V:.3e}  feas {feas}  feasRes {feasRes:.3g}  '
              f'Time(s) {elapsed:.2f}  gradnorm {gn:.3g}  '
              f'funCalls {simCount["n"]}')
        logRow(iterIdx, k, Ts_h, M_h, SSe_h,
               Ts_V, M_V, SSe_V, feas, feasRes, elapsed, gn)
        savePlot(k, idx=iterIdx, data=data)

    res = minimize(fun, k0,
                   method='SLSQP',
                   jac=None,
                   constraints=slsqpCons,
                   bounds=bnds,
                   callback=callback,
                   options={'maxiter': 200,
                            'ftol': 1e-5,
                            'disp': True,
                            'eps': 1.4901161193847656e-08})

    print('solve done')
    print('best gains vector:', res.x)

    # ensure final iterate is logged/plotted if different from last callback
    if lastLogged['k'] is None or not np.allclose(res.x, lastLogged['k'],
                                                  rtol=1e-12, atol=1e-12):
        kFinal = np.asarray(res.x, float)
        data = once(kFinal, count=False)
        (_, _, _, _, _,
         Ts_h, M_h, SSe_h, Ts_V, M_V, SSe_V, ok) = data
        feas = (M_h <= 0.40 and abs(SSe_h) <= 1e-3 and
                Ts_V <= 20.0 and M_V <= 0.40 and abs(SSe_V) <= 1e-3)
        elapsed = time.time() - startTime
        gn = gradNormEst(kFinal, f0=Ts_h)
        feasRes = feasibilityResidual(M_h, SSe_h, M_V, SSe_V, Ts_V)
        iterIdx = lastLogged['idx'] + 1
        logRow(iterIdx, kFinal, Ts_h, M_h, SSe_h,
               Ts_V, M_V, SSe_V, feas, feasRes, elapsed, gn)
        savePlot(kFinal, idx=iterIdx, data=data)

    print('\nSLSQP details:')
    print(' status:', getattr(res, 'status', None))
    print(' message:', getattr(res, 'message', None))
    print(' niter:', getattr(res, 'nit', None))
    print(' nfev:', getattr(res, 'nfev', None))

    return res


if __name__ == "__main__":
    # start from your hand-tuned gains as initial guess
    initK = np.array([0.04, 0.01,   # Kp_h, Ki_h
                      2.0, 0.6, 0.25, 20.0,  # Kp_th, Ki_th, Kd_th, N_th
                      0.3, 0.08,    # Kp_V, Ki_V
                      5.0, 0.0])    # Kaw, Kff_gam
    runOptimization(initK)
