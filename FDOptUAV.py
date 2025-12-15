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
print("Python: connected to MATLAB engine 'engine_1'", flush=True)
model = 'uav_problem'  # Simulink model name

eng.load_system(model, nargout=0)
eng.set_param(model, 'SimulationCommand', 'stop', nargout=0)
eng.set_param(model, 'SimulationMode', 'accelerator', nargout=0) #SPEEEEEEEEEEEEED
eng.set_param(model, 'SolverType', 'Variable-step', nargout=0)
# Use a lighter, faster nonstiff solver and looser tolerances for speed
eng.set_param(model, 'Solver', 'ode45', nargout=0)
eng.set_param(model, 'StopTime', '30', nargout=0)
eng.set_param(model, 'RelTol', '1e-4', nargout=0)
eng.set_param(model, 'AbsTol', '1e-6', nargout=0)
eng.set_param(model, 'MaxStep', '1e-3', nargout=0) # was 5e-3 reduced for Ts better accuracy.

# Design vector order:
# k[0] = Kp_h, k[1] = Ki_h,
# k[2] = Kp_th, k[3] = Ki_th, k[4] = Kd_th,
# k[5] = Kp_V, k[6] = Ki_V
DESIGN_NAMES = ["Kp_h", "Ki_h", "Kp_th", "Ki_th",
                "Kd_th", "Kp_V", "Ki_V"]


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

    eng.workspace['Kp_h']   = Kp_h
    eng.workspace['Ki_h']   = Ki_h

    eng.workspace['Kp_th']  = Kp_th
    eng.workspace['Ki_th']  = Ki_th
    eng.workspace['Kd_th']  = Kd_th
    eng.workspace['N_th']   = N_th

    eng.workspace['Kp_V']   = Kp_V
    eng.workspace['Ki_V']   = Ki_V

    # --- Simulation settings ---
    tstop = 30.0
    eng.workspace['tstop'] = tstop


# initialize parameters in MATLAB
init_uav_params_in_workspace(eng)


def fmtVal(val):
    return f"{val:.6g}".replace('+', '').replace(' ', '')


def buildLabel(k):
    return ("Kph" + fmtVal(k[0]) +
            "_Kih" + fmtVal(k[1]) +
            "_Kpth" + fmtVal(k[2]) +
            "_Kith" + fmtVal(k[3]) +
            "_Kdth" + fmtVal(k[4]) +
            "_KpV" + fmtVal(k[5]) +
            "_KiV" + fmtVal(k[6]))


def toVec(x):
    return np.asarray(x).squeeze()

def _expand_ref_to_time(t, r):
    """Return a reference vector the same length as t.

    Handles scalar refs or short vectors (e.g., step values like [r0, r1])
    by broadcasting the last value across the full time vector.
    If lengths already match, returns r unchanged.
    """
    tt = np.asarray(t).ravel()
    rr = np.asarray(r).squeeze()
    if rr.ndim == 0 or rr.size == 1:
        return np.full_like(tt, float(rr))
    if rr.size != tt.size:
        return np.full_like(tt, float(rr[-1]))
    return rr


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
    Delta-based bands from output: delta = y_ss - y_init; band = y_init + delta*(1±pct).
    Overshoot relative to delta; SSe uses reference target minus output steady-state.
    """
    if t.size == 0 or y.size == 0:
        return np.inf, np.inf, np.inf

    t_vec = toVec(t)
    y_vec = toVec(y)
    if t_vec.size < 2:
        return np.inf, np.inf, np.inf
    # dense uniform resampling to tighten crossing interpolation
    n_dense = max(2000, t_vec.size * 5)
    t_dense = np.linspace(t_vec[0], t_vec[-1], n_dense)
    y_dense = np.interp(t_dense, t_vec, y_vec)
    # reference resample if vector
    if np.size(r) > 1:
        r_vec = toVec(r)
        r_dense = np.interp(t_dense, t_vec, r_vec)
        r_target = float(r_dense[-1])
    else:
        r_target = float(r)
        r_dense = np.full_like(t_dense, r_target)

    y_init = float(y_dense[0])
    y_ss = tailValue(y_dense, max(10, y_dense.size // 10))
    delta = y_ss - y_init

    # If essentially no change, fall back to original definitions
    if abs(delta) < 1e-12:
        M = (np.max(y_dense) - y_ss) / (y_ss if y_ss != 0 else 1.0)
        Ts = settle(t_dense, y_dense, y_ss, pct=pct_band)
        SSe = r_target - y_ss
        return Ts, M, SSe

    band_hi = y_init + delta * (1.0 + pct_band)
    band_lo = y_init + delta * (1.0 - pct_band)
    band_min = min(band_lo, band_hi)
    band_max = max(band_lo, band_hi)

    def settle_delta(tt, yy, lo, hi):
        if tt.size == 0:
            return np.inf
        ok = (yy >= lo) & (yy <= hi)
        stay = ok.copy()
        for i in range(len(stay) - 2, -1, -1):
            stay[i] = stay[i] and stay[i + 1]
        i0 = np.argmax(stay)
        if not stay[i0]:
            return np.inf
        j = i0
        while j > 0 and lo <= yy[j - 1] <= hi:
            j -= 1
        if j == 0:
            return float(tt[i0])
        e1 = yy[j - 1]
        e2 = yy[j]
        if e1 == e2:
            return float(tt[j])
        # interpolate toward the band edge in the direction of delta
        target_edge = hi if delta > 0 else lo
        alpha = (target_edge - e1) / (e2 - e1)
        tcross = tt[j - 1] + alpha * (tt[j] - tt[j - 1])
        return float(tcross)

    Ts = settle_delta(t_dense, y_dense, band_min, band_max)
    M = (np.max(y_dense) - y_ss) / abs(delta)
    SSe = r_target - y_ss
    return Ts, M, SSe


def feasibilityResidual(M_h, SSe_h, SSe_V, vmax):
    """
    Max violation of constraints:
    M_h <= 0.10
    |SSe_h| <= 1e-3
    V max <= 22
    |SSe_V| <= 1e-3
    """
    viol = 0.0
    viol = max(viol, M_h - 0.10)
    viol = max(viol, abs(SSe_h) - 1e-3)
    viol = max(viol, vmax - 22.0)
    viol = max(viol, abs(SSe_V) - 1e-3)
    return viol


def runOptimization(initK, baseDir=None):
    k0 = np.asarray(initK, float)
    print(f"Python: runOptimization starting, k0={k0}", flush=True)
    label = buildLabel(k0)
    outDir = baseDir if baseDir else f"slicesUAV_{label}"
    os.makedirs(outDir, exist_ok=True)
    logPath = os.path.join(outDir, f"opt_log_{label}.csv")

    startTime = time.time()
    initK0 = k0.copy()
    simCount = {'n': 0}
    evalCount = {'n': 0}
    lastLogged = {'idx': 0, 'k': None}

    # CSV header
    with open(logPath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['SLSQPiter'] +
            DESIGN_NAMES +
            ['Ts_h', 'M_h', 'SSe_h', 'SSe_V', 'Vmax',
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
            eng.workspace['N_th']    = 1.0  # fixed filter coefficient
            eng.workspace['Kp_V']    = float(k[5])
            eng.workspace['Ki_V']    = float(k[6])

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
            # Ensure references align with time for metrics/plots
            h_ref  = _expand_ref_to_time(t, h_ref)
            V_ref  = _expand_ref_to_time(t, V_ref)
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
        # For velocity, only steady-state error is used; overshoot/Ts not constrained
        _, _, SSe_V = step_metrics(t, V_out, V_ref)
        vmax = float(np.max(V_out)) if V_out.size > 0 else np.inf

        if not np.isfinite(Ts_h):
            return (t, h_out, V_out, h_ref, V_ref,
                    np.inf, np.inf, np.inf, np.inf, np.inf, False)

        return (t, h_out, V_out, h_ref, V_ref,
                Ts_h, M_h, SSe_h, SSe_V, vmax, True)

    def once(k, count=False):
        return saferun(k, count=count)

    def evalCostSilent(k):
        *_, Ts_h, M_h, SSe_h, SSe_V, vmax, ok = once(k, count=False)
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

    def logRow(iterIdx, k, Ts_h, M_h, SSe_h, SSe_V, vmax,
               feas, feasRes, elapsed, gn):
        with open(logPath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [iterIdx] +
                [float(val) for val in k] +
                [Ts_h, M_h, SSe_h, SSe_V, vmax,
                 feas, feasRes, elapsed, gn, simCount['n']]
            )
        lastLogged['idx'] = iterIdx
        lastLogged['k'] = np.asarray(k, float)

    def savePlot(k, idx=None, data=None):
        if data is None:
            t, h_out, V_out, h_ref, V_ref, Ts_h, M_h, SSe_h, SSe_V, vmax, ok = once(k, count=False)
        else:
            t, h_out, V_out, h_ref, V_ref, Ts_h, M_h, SSe_h, SSe_V, vmax, ok = data

        # Guard: ensure refs are time-aligned even if provided via cached 'data'
        try:
            if t is not None:
                h_ref = _expand_ref_to_time(t, h_ref)
                V_ref = _expand_ref_to_time(t, V_ref)
        except Exception:
            pass

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        # Height subplot
        if t is not None and h_out is not None:
            axes[0].plot(t, h_out, label='h')
            axes[0].plot(t, h_ref, '--', label='h_ref')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Height')
            axes[0].grid(True)
            axes[0].set_box_aspect(1)
            if h_out is not None and h_out.size > 1:
                h_init = float(h_out[0])
                h_ss = tailValue(h_out, max(10, h_out.size // 10))
                delta_h = h_ss - h_init
                if abs(delta_h) > 0:
                    axes[0].axhline(h_init + delta_h * 1.02, color='gray', linestyle=':', linewidth=1.2,
                                    label='h +/-2% band')
                    axes[0].axhline(h_init + delta_h * 0.98, color='gray', linestyle=':', linewidth=1.2)
                    axes[0].axhline(h_init + delta_h * 1.10, color='red', linestyle='--', linewidth=1.2,
                                    label='h 10% OS')
                y_vals = [np.min(h_out), np.max(h_out),
                          h_init + delta_h * 1.02, h_init + delta_h * 0.98, h_init + delta_h * 1.10]
                y_margin = 0.05 * max(1.0, np.ptp(y_vals))
                axes[0].set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

        # Velocity subplot
        if t is not None and V_out is not None:
            axes[1].plot(t, V_out, label='V')
            axes[1].plot(t, V_ref, '--', label='V_ref')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Velocity')
            axes[1].grid(True)
            axes[1].set_box_aspect(1)
            if V_out is not None and V_out.size > 1:
                axes[1].axhline(22.0, color='red', linestyle='--', linewidth=1.2, label='V max 22')
                y_vals = [np.min(V_out), np.max(V_out), 22.0]
                y_margin = 0.05 * max(1.0, np.ptp(y_vals))
                axes[1].set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

        # Combine legends and place outside to the right with initial gains noted
        handles, labels_ = [], []
        for ax in axes:
            h_ax, l_ax = ax.get_legend_handles_labels()
            handles += h_ax
            labels_ += l_ax
        handles.append(plt.Line2D([], [], color='none',
                                   label=f'Ts_h={Ts_h:.3g}, M_h={M_h:.3g}, SSe_h={SSe_h:.2e}'))
        labels_.append(f'Ts_h={Ts_h:.3g}, M_h={M_h:.3g}, SSe_h={SSe_h:.2e}')
        handles.append(plt.Line2D([], [], color='none',
                                   label=f'SSe_V={SSe_V:.2e}, Vpeak={vmax:.3g}'))
        labels_.append(f'SSe_V={SSe_V:.2e}, Vpeak={vmax:.3g}')
        axes[1].legend(handles, labels_, loc='upper left', bbox_to_anchor=(1.15, 1.0))

        plotIdx = lastLogged['idx'] if idx is None else idx
        fig.suptitle(f'iter_{plotIdx:03d}')
        fig.subplots_adjust(left=0.01, right=0.7, wspace=0.01, top=0.9, bottom=0.12)
        name = os.path.join(outDir, f'iter{plotIdx:03d}_{label}.pdf')
        fig.savefig(name)
        plt.close(fig)

    def fun(k):
        t0 = time.time()
        t, h_out, V_out, h_ref, V_ref, Ts_h, M_h, SSe_h, SSe_V, vmax, ok = once(k, count=True)
        elapsed = time.time() - t0
        evalCount['n'] += 1
        calls = simCount['n']
        if not ok:
            print(f'FDeval {evalCount["n"]:04d}: k={k}  unstable/bad -> large cost  '
                  f'Time(s) {elapsed:.2f}  funCalls {calls}')
            return 1e6
        print(f'FDeval {evalCount["n"]:04d}: k={k}  Ts_h {Ts_h:.6g}  M_h {M_h:.6g}  '
              f'SSe_h {SSe_h:.3e}  SSe_V {SSe_V:.3e}  Vmax {vmax:.3g}  '
              f'Time(s) {elapsed:.2f}  funCalls {calls}')
        return Ts_h  # objective

    def consVals(k):
        _, _, _, _, _, Ts_h, M_h, SSe_h, SSe_V, vmax, ok = once(k, count=True)
        if not ok or not np.isfinite(M_h) or not np.isfinite(SSe_h):
            return np.array([-1.0] * 6, float)

        # c >= 0 style inequalities
        c1 = 0.10 - M_h         # height overshoot <= 10%
        c2 = 1e-3 - SSe_h       # SSe_h <= 1e-3
        c3 = 1e-3 + SSe_h       # -SSe_h <= 1e-3
        c4 = 22.0 - vmax        # velocity must stay below 22
        c5 = 1e-3 - SSe_V       # SSe_V <= 1e-3
        c6 = 1e-3 + SSe_V       # -SSe_V <= 1e-3
        return np.array([c1, c2, c3, c4, c5, c6], float)

    # Bounds: simple -10..10 on each design variable
    bnds = Bounds(lb=[-10.0] * len(k0), ub=[10.0] * len(k0))

    def consFun(k):
        return consVals(k)

    slsqpCons = [
        {'type': 'ineq', 'fun': (lambda k, idx=i: consFun(k)[idx])}
        for i in range(6)
    ]

    # Initial iterate metrics / log
    print("Python: starting initial simulation for k0", flush=True)
    initData = once(k0, count=False)
    (initT, initH, initV, initHref, initVref,
     initTs_h, initM_h, initSSe_h, initSSe_V, initVmax, initOk) = initData

    initFeas = (initM_h <= 0.10 and abs(initSSe_h) <= 1e-3 and
                initVmax <= 22.0 and abs(initSSe_V) <= 1e-3)
    initFeasRes = feasibilityResidual(initM_h, initSSe_h, initSSe_V, initVmax)
    initElapsed = time.time() - startTime
    initGradNorm = gradNormEst(k0, f0=initTs_h)
    print(f"Python: initial simulation done (ok={initOk})  Ts_h={initTs_h:.3g}", flush=True)
    logRow(0, k0, initTs_h, initM_h, initSSe_h,
           initSSe_V, initVmax, initFeas, initFeasRes,
           initElapsed, initGradNorm)
    savePlot(k0, idx=0, data=initData)
    lastLogged['k'] = np.asarray(k0, float)
    print('Python: start solve (SLSQP)', flush=True)

    def callback(k, state=None):
        if lastLogged['k'] is not None and np.allclose(k, lastLogged['k'],
                                                      rtol=1e-12, atol=1e-12):
            return
        data = once(k, count=False)
        (_, _, _, _, _,
         Ts_h, M_h, SSe_h, SSe_V, vmax, ok) = data
        feas = (M_h <= 0.10 and abs(SSe_h) <= 1e-3 and
                vmax <= 22.0 and abs(SSe_V) <= 1e-3)
        elapsed = time.time() - startTime
        gn = gradNormEst(k, f0=Ts_h)
        feasRes = feasibilityResidual(M_h, SSe_h, SSe_V, vmax)
        iterIdx = lastLogged['idx'] + 1
        print(f'SLSQPiter {iterIdx:03d}: k={k}  Ts_h {Ts_h:.5g}  M_h {M_h:.5g}  '
              f'SSe_h {SSe_h:.3e}  SSe_V {SSe_V:.3e}  Vmax {vmax:.3g}  '
              f'feas {feas}  feasRes {feasRes:.3g}  '
              f'Time(s) {elapsed:.2f}  gradnorm {gn:.3g}  '
              f'funCalls {simCount["n"]}')
        logRow(iterIdx, k, Ts_h, M_h, SSe_h,
               SSe_V, vmax, feas, feasRes, elapsed, gn)
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
                            'eps': 1.4901161193847656e-06}) #was 1.4901161193847656e-08 

    print('solve done')
    print('best gains vector:', res.x)

    # ensure final iterate is logged/plotted if different from last callback
    if lastLogged['k'] is None or not np.allclose(res.x, lastLogged['k'],
                                                  rtol=1e-12, atol=1e-12):
        kFinal = np.asarray(res.x, float)
        data = once(kFinal, count=False)
        (_, _, _, _, _,
         Ts_h, M_h, SSe_h, SSe_V, vmax, ok) = data
        feas = (M_h <= 0.10 and abs(SSe_h) <= 1e-3 and
                vmax <= 21.0 and abs(SSe_V) <= 1e-3)
        elapsed = time.time() - startTime
        gn = gradNormEst(kFinal, f0=Ts_h)
        feasRes = feasibilityResidual(M_h, SSe_h, SSe_V, vmax)
        iterIdx = lastLogged['idx'] + 1
        logRow(iterIdx, kFinal, Ts_h, M_h, SSe_h,
               SSe_V, vmax, feas, feasRes, elapsed, gn)
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
                      2.0, 0.6, 0.25,  # Kp_th, Ki_th, Kd_th
                      0.3, 0.08])    # Kp_V, Ki_V
    runOptimization(initK)
