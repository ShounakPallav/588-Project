# --- Python <-> Simulink PID tuning with analytic gradients ------------------
import os
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import matlab.engine  # type: ignore

# ---------------------------------------------------------------------------
# user knobs
MODEL    = 'ProofofConceptProblem'   # <-- your model name (no .slx)
STOPTIME = '100'                      # string for MATLAB set_param
BOUNDS   = [(-100.0, 100.0), (-100.0, 100.0), (-50.0, 50.0)]  # [kp, ki, kd]
X0       = np.array([0.5, 1.0, 0.05])                          # initial guess
SAVE_FIGS = False
SSE_TOL   = 1e-3
USE_ANALYTIC = True
DO_GRAD_CHECK = True
# Drive gains via base workspace variables only (kp, ki, kd)
# If True: do NOT overwrite PID/Constant blocks numerically; rely on 'kp','ki','kd' in model
WORKSPACE_ONLY = True
# Optional: explicit Constant block paths to treat as [kp, ki, kd]
# Provided by user
SENS_CONST_HINTS = []
# ---------------------------------------------------------------------------

def ensure_Nx3(raw):
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[1] == 3:        # (N,3)
        return arr
    if arr.ndim == 3 and arr.shape[:2] == (3,1):   # (3,1,N)
        return np.transpose(arr, (2,0,1)).reshape(arr.shape[2], 3)
    if arr.ndim == 3 and arr.shape[:2] == (1,3):   # (1,3,N)
        return np.transpose(arr, (2,1,0)).reshape(arr.shape[2], 3)
    raise ValueError(f"Unexpected ek shape: {arr.shape}")

def smooth_abs(x, eps=1e-4):
    return np.sqrt(x*x + eps*eps)

# --- metrics ----------------------------------------------------------------

def settling_time(t, y, r, tol=0.05):
    """Return 5% settling time wrt final reference value.
    Returns np.inf if never within band for the remainder of the horizon."""
    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    r = np.asarray(r).ravel()
    if len(t) == 0:
        return np.inf
    # robust final value: median of last 10% samples
    tail = max(1, len(r)//10)
    rf = np.median(r[-tail:])
    band = 0.05 * max(abs(rf), 1.0)
    err = np.abs(y - rf)
    if err[-1] > band:
        return np.inf
    idx = np.where(err > band)[0]
    if len(idx) == 0:
        return t[0]
    j = idx[-1] + 1
    if j >= len(t):
        return np.inf
    return t[j]

# --- objective & constraints (all differentiable) ---------------------------


def overshoot_val_grad(t, y, eK, r, alpha=60.0):
    """Smooth overshoot as fraction of final reference magnitude."""
    r = np.asarray(r).ravel()
    y = np.asarray(y).ravel()
    tail = max(1, len(r)//10)
    rf = np.median(r[-tail:])
    ar = max(abs(rf), 1.0)
    z = y - rf
    m = np.max(z)
    w = np.exp(alpha*(z - m))
    S = w.sum()
    sm = m + np.log(S/len(z))/alpha  # smooth max
    OS = sm / ar
    dsm_dy = w/S
    dOS = (dsm_dy[:,None]/ar * (-eK)).sum(axis=0)  # dy/dK = -de/dK
    return OS, dOS

def sse_val_grad(t, e, eK, W=1.0):
    T = t[-1]
    mask = t >= (T - W)
    tw = t[mask]
    dt = np.diff(tw, prepend=tw[0])
    ew = e[mask]
    E2 = (dt * (ew**2)).sum() / max(W, 1e-12)
    SSE = np.sqrt(max(E2, 1e-16))
    dE2 = (2.0/max(W,1e-12)) * (dt[:,None] * (ew[:,None] * eK[mask,:])).sum(axis=0)
    dSSE = 0.5 * dE2 / SSE
    return SSE, dSSE

def itae_val_grad(t, e, eK, eps=1e-4):
    t = np.asarray(t).ravel()
    e = np.asarray(e).ravel()
    dt = np.diff(t, prepend=t[0])
    J = np.sum(dt * (t * smooth_abs(e, eps)))
    sigma = e / smooth_abs(e, eps)
    gJ = np.sum(dt[:,None] * (t[:,None] * (sigma[:,None] * eK)), axis=0)
    return J, gJ

def itae_value_only(t, e, eps=1e-4):
    t = np.asarray(t).ravel()
    e = np.asarray(e).ravel()
    dt = np.diff(t, prepend=t[0])
    return float(np.sum(dt * (t * smooth_abs(e, eps))))


# --- Simulink runner --------------------------------------------------------
class SimRunner:
    def __init__(self, model, stoptime='10'):
        # start MATLAB and prepare the model
        self.eng = matlab.engine.start_matlab()
        self.eng.load_system(model, nargout=0)
        # Use full rebuild between sims to ensure mask/workspace params refresh
        self.eng.set_param(model, 'FastRestart', 'off', nargout=0)
        self.eng.set_param(model, 'StopTime', stoptime, nargout=0)
        self.model = model
        # location to save any figures: Project/slices
        self.slices_dir = os.path.join(os.getcwd(), 'slices')
        os.makedirs(self.slices_dir, exist_ok=True)

        # --- ensure time is logged and a SimulationOutput is returned
        self.eng.set_param(model, 'SaveTime', 'on', 'TimeSaveName', 'tout', nargout=0)
        self.eng.set_param(model, 'ReturnWorkspaceOutputs', 'on', nargout=0)    
        self.eng.set_param(model, 'RelTol', '1e-6', 'AbsTol', '1e-8', nargout=0)

        # fixed params the model expects
        self.eng.workspace['N'] = float(20.0)
        # Realize P(s) = (s-1)/(s^2+2s+6) in state-space for the model
        self.eng.eval("[A,B,C,D] = tf2ss([1 1],[1 2 6]);", nargout=0) #s+1 numerator
        # discover PID Controller blocks once
        self.pid_blocks = []
        try:
            # Use a valid function handle for MatchFilter via MATLAB engine
            mf_all = self.eng.eval('@Simulink.match.allVariants', nargout=1)
            p1 = self.eng.find_system(self.model,
                                      'MatchFilter', mf_all,
                                      'FollowLinks', 'on', 'LookUnderMasks', 'all',
                                      'MaskType', 'PID Controller')
            p2 = self.eng.find_system(self.model,
                                      'MatchFilter', mf_all,
                                      'FollowLinks', 'on', 'LookUnderMasks', 'all',
                                      'MaskType', 'PID 1dof')
            cands = []
            if p1:
                cands += list(p1)
            if p2:
                cands += list(p2)
            self.pid_blocks = [str(b) for b in cands]
            if self.pid_blocks:
                print('   Found PID blocks:')
                for b in self.pid_blocks:
                    print('     - ' + b)
            else:
                print('   No PID Controller blocks discovered via MaskType search.')
        except Exception as ex:
            print('   (warn) PID block discovery failed: ' + str(ex))

    def _echo_base_gains(self):
        try:
            kp = float(self.eng.workspace['kp'])
            ki = float(self.eng.workspace['ki'])
            kd = float(self.eng.workspace['kd'])
            print(f"   base kp={kp:.6g}  ki={ki:.6g}  kd={kd:.6g}")
        except Exception:
            print('   (warn) base kp/ki/kd not present')

    def _echo_var_resolution(self):
        try:
            self.eng.eval(
                "disp('   Variable resolution (Simulink.findVars):');"
                f"vars_kp = Simulink.findVars('{self.model}','Name','kp');"
                f"vars_ki = Simulink.findVars('{self.model}','Name','ki');"
                f"vars_kd = Simulink.findVars('{self.model}','Name','kd');"
                "if isempty(vars_kp), disp('     kp: <none>'); else, disp(vars_kp); end;"
                "if isempty(vars_ki), disp('     ki: <none>'); else, disp(vars_ki); end;"
                "if isempty(vars_kd), disp('     kd: <none>'); else, disp(vars_kd); end;",
                nargout=0
            )
        except Exception:
            pass

    def _set_pid_gains(self, kp, ki, kd):
        if not self.pid_blocks:
            return
        for blk in self.pid_blocks:
            try:
                # break link if needed, then set numeric gains
                try:
                    self.eng.set_param(blk, 'LinkStatus', 'inactive', nargout=0)
                except Exception:
                    pass
                self.eng.set_param(blk, 'P', str(kp), 'I', str(ki), 'D', str(kd), nargout=0)
            except Exception as ex:
                print("   (warn) set_param P/I/D on '" + blk + "' failed: " + str(ex))

    def _set_sens_consts(self, kp, ki, kd):
        """Set gains used by sensitivity/derivative paths.

        Strategy:
        1) Prefer Constant blocks whose Value is the literal 'kp','ki','kd' (case‑sensitive),
           including inside Variant Subsystems.
        2) If none found, fall back to any Constant blocks under subsystems whose path contains
           'Sensitiv' and whose names look like Kp/Ki/Kd (case‑insensitive).
        """
        try:
            # If user provided explicit block paths, use them first
            if SENS_CONST_HINTS and len(SENS_CONST_HINTS) == 3:
                try:
                    self.eng.set_param(SENS_CONST_HINTS[0], 'Value', str(kp), nargout=0)
                    self.eng.set_param(SENS_CONST_HINTS[1], 'Value', str(ki), nargout=0)
                    self.eng.set_param(SENS_CONST_HINTS[2], 'Value', str(kd), nargout=0)
                    return
                except Exception:
                    # fall through to discovery
                    pass

            # Include blocks inside Variant Subsystems using a function handle MatchFilter
            mf_all = self.eng.eval('@Simulink.match.allVariants', nargout=1)
            common = ['MatchFilter', mf_all, 'FollowLinks', 'on', 'LookUnderMasks', 'all',
                      'BlockType', 'Constant']

            kp_blks = self.eng.find_system(self.model, *common, 'Value', 'kp')
            ki_blks = self.eng.find_system(self.model, *common, 'Value', 'ki')
            kd_blks = self.eng.find_system(self.model, *common, 'Value', 'kd')

            set_count = 0
            for blk in list(kp_blks):
                self.eng.set_param(blk, 'Value', str(kp), nargout=0)
                set_count += 1
            for blk in list(ki_blks):
                self.eng.set_param(blk, 'Value', str(ki), nargout=0)
                set_count += 1
            for blk in list(kd_blks):
                self.eng.set_param(blk, 'Value', str(kd), nargout=0)
                set_count += 1

            # Heuristic fallback: look under any subsystem path that includes 'Sensitiv'
            # and try to match blocks by name (Kp/Ki/Kd, case-insensitive)
            if set_count == 0:
                all_consts = list(self.eng.find_system(self.model, *common))
                if all_consts:
                    for blk in all_consts:
                        try:
                            parent = str(self.eng.get_param(blk, 'Parent'))
                            name   = str(self.eng.get_param(blk, 'Name'))
                            path   = parent + '/' + name
                            if 'sensitiv' in path.lower():
                                lname = name.lower()
                                if 'kp' == lname or lname.endswith('kp') or 'k_p' in lname:
                                    self.eng.set_param(blk, 'Value', str(kp), nargout=0)
                                    set_count += 1
                                elif 'ki' == lname or lname.endswith('ki') or 'k_i' in lname:
                                    self.eng.set_param(blk, 'Value', str(ki), nargout=0)
                                    set_count += 1
                                elif 'kd' == lname or lname.endswith('kd') or 'k_d' in lname:
                                    self.eng.set_param(blk, 'Value', str(kd), nargout=0)
                                    set_count += 1
                        except Exception:
                            pass

            if set_count == 0:
                print("   (warn) No sensitivity constants updated. Ensure your Constant blocks use 'kp','ki','kd' as Value or reside under a '*Sensitiv*' subsystem with Kp/Ki/Kd names.")
        except Exception as ex:
            print('   (warn) set_sens_consts failed: ' + str(ex))

    def _echo_sens_consts(self):
        try:
            mf_all = self.eng.eval('@Simulink.match.allVariants', nargout=1)
            allc = self.eng.find_system(self.model,
                                        'MatchFilter', mf_all,
                                        'FollowLinks', 'on', 'LookUnderMasks', 'all',
                                        'BlockType', 'Constant')
            for blk in list(allc):
                try:
                    v = self.eng.get_param(blk, 'Value')
                    if v in ('kp','ki','kd') or v.replace('.','',1).replace('-','',1).isdigit():
                        print(f"   Const '{blk}': Value={v}")
                except Exception:
                    pass
        except Exception:
            pass

    def _echo_pid_gains(self):
        if not self.pid_blocks:
            return
        for blk in self.pid_blocks:
            try:
                P = self.eng.get_param(blk, 'P')
                I = self.eng.get_param(blk, 'I')
                D = self.eng.get_param(blk, 'D')
                print("   PID '" + blk + "': P=" + str(P) + " I=" + str(I) + " D=" + str(D))
            except Exception as ex:
                print("   (warn) get_param P/I/D on '" + blk + "' failed: " + str(ex))

    def simulate(self, K):
        # 1) workspace variables for any expressions
        self.eng.workspace['kp'] = float(K[0])
        self.eng.workspace['ki'] = float(K[1])
        self.eng.workspace['kd'] = float(K[2])
        # echo base workspace values and variable resolution
        self._echo_base_gains()
        self._echo_var_resolution()
        # 2) optionally set PID mask and Sensitivities constants numerically
        if not WORKSPACE_ONLY:
            self._set_pid_gains(float(K[0]), float(K[1]), float(K[2]))
            self._set_sens_consts(float(K[0]), float(K[1]), float(K[2]))
        try:
            self.eng.set_param(self.model, 'SimulationCommand', 'update', nargout=0)
        except Exception:
            pass
        # echo active PID gains and consts
        self._echo_pid_gains()
        self._echo_sens_consts()
        # 3) run sim and capture SimulationOutput in 'out'
        self.eng.eval(f"out = sim('{self.model}');", nargout=0)
        # 4) ensure 'tout' exists in base workspace
        self.eng.eval("if ~exist('tout','var') && exist('out','var'); tout = out.tout; end", nargout=0)
        self.eng.eval("if ~exist('y','var') && exist('out','var'); try, y = out.y; catch, end; if ~exist('y','var'); try, y = out.get('y'); catch, end; end; end", nargout=0)
        self.eng.eval("if ~exist('ek','var') && exist('out','var'); try, ek = out.ek; catch, end; if ~exist('ek','var'); try, ek = out.get('ek'); catch, end; end; end", nargout=0)
        self.eng.eval("if ~exist('r','var') && exist('out','var'); try, r = out.r; catch, end; if ~exist('r','var'); try, r = out.get('r'); catch, end; end; end", nargout=0)

        t  = np.asarray(self.eng.workspace['tout']).ravel()
        y  = np.asarray(self.eng.workspace['y']).ravel()
        ek = ensure_Nx3(self.eng.workspace['ek'])
        try:
            r = np.asarray(self.eng.workspace['r']).ravel()
        except Exception:
            r = np.ones_like(y)
        # keep full trajectories; lengths may differ slightly (value uses full t/y/r)
        return t, y, r, ek

    def save_plot(self, idx, J, OS, SSE):
        if not SAVE_FIGS:
            return
        out = os.path.join(
            self.slices_dir,
            f"{self.model}_sim{idx:03d}_J{J:.4g}_OS{OS*100:.2f}_SSE{SSE:.2e}.pdf"
        ).replace('\\', '/')
        cmd = (
            "f=figure('Visible','off');"
            "plot(tout,r,'k--','LineWidth',1.25); hold on;"
            "plot(tout,y,'b','LineWidth',1.5); grid on;"
            "xlabel('t'); title('Reference r(t) and Output y(t)');"
            "legend(''r(t)'',''y(t)'',''Location'',''best''); hold off;"
            f"exportgraphics(f,'{out}','Resolution',150);"
            "close(f);"
        )
        try:
            self.eng.eval(cmd, nargout=0)
            print(f"   saved plot -> {out}")
        except Exception as ex:
            print(f"   (plot save skipped: {ex})")

    def close(self):
        try:
            self.eng.set_param(self.model, 'FastRestart', 'off', nargout=0)
        except Exception:
            pass
        try:
            self.eng.close_system(self.model, 0, nargout=0)
        except Exception:
            pass
        try:
            self.eng.quit()
        except Exception:
            pass

# --- custom trust-constr callback -------------------------------------------
def cb_trust(xk, state=None):
    def get(st, names, default=float('nan')):
        for nm in names:
            if hasattr(st, nm):
                val = getattr(st, nm)
                if val is not None:
                    return val
        return default
    niter = int(get(state, ['niter','nit','iter'], -1))
    nfev  = int(get(state, ['nfev','f_evals','fun_evals','func_evals'], -1))
    cg    = int(get(state, ['cg_niter','cg_iter','cg_iters'], -1))
    f     = float(get(state, ['objective','f','fun'], float('nan')))
    tr    = float(get(state, ['tr_radius','trust_radius'], float('nan')))
    opt   = float(get(state, ['optimality','kkt','kkt_residual'], float('nan')))
    viol  = float(get(state, ['constr_violation','c_viol'], float('nan')))
    pen   = float(get(state, ['penalty','merit_weight'], float('nan')))
    mu    = float(get(state, ['barrier_parameter','barrier_param'], float('nan')))
    cgs   = int(get(state, ['cg_stop','cg_stop_cond'], -1))
    if niter == 1 or (niter % 5) == 0:
        print('|       | Total Func Evals| CG step| Objective Func Value | step size | Optimality | Feasibility | Optimality + Feasibility | interior t  | reason |')
    print(f"| {niter:5d} | {nfev:7d} | {cg:6d} | {f:+.4e} | {tr:8.2e} | {opt:8.2e} | {viol:8.2e} | {pen:8.2e} |  {mu:9.2e}  | {cgs:5d} |")

# --- Cached evaluator so each point simulates exactly once ------------------

def cb_repeat_header(xk, state=None):
    try:
        niter = getattr(state, 'niter', None)
        if niter is None:
            niter = getattr(state, 'nit', None)
    except Exception:
        niter = None
    if niter is not None and niter > 0 and (niter % 5) == 0:
        print('| niter |f evals|CG iter|  obj func   |tr radius |   opt    |  c viol  | penalty  |barrier param|CG stop|')


def finite_diff_grad(runner, K, tol=0.05, h=1e-4, rel=1e-1):
    import numpy as _np
    K = _np.asarray(K, dtype=float)
    g = _np.zeros(3)
    for i in range(3):
        # use a relative step based on `rel` (fallback to absolute `h`)
        hi = max(h, rel*abs(K[i]) if abs(K[i]) > 0 else h)
        Kp = K.copy(); Kp[i] += hi    
        Km = K.copy(); Km[i] -= hi
        t, y, r, _ = runner.simulate(Kp); e = r - y
        Jp = itae_value_only(t, e)
        t, y, r, _ = runner.simulate(Km); e = r - y
        Jm = itae_value_only(t, e)
        g[i] = (Jp - Jm) / (2*hi)
    return g

def check_gradients(runner, K, tol=0.05, h=1e-4, rel=1e-1):
    import numpy as _np
    t, y, r, ek = runner.simulate(K); e = r - y
    J, dJ = itae_val_grad(t, e, ek)
    gfd = finite_diff_grad(runner, K, tol=tol, h=h, rel=rel)
    err = _np.linalg.norm(dJ - gfd) / max(1e-12, _np.linalg.norm(gfd))
    print("\nGradient check at K =", K)
    print(" analytic dJ =", dJ)
    print(" numeric  dJ =", gfd)
    print(" relative error =", err)

class Evaluator:
        def __init__(self, runner):
            self.runner = runner
            self.last_x = None
            self.last   = None
            self.count  = 0
    
        def evaluate(self, K):
            K = np.asarray(K, dtype=float)
            if self.last_x is not None and np.allclose(K, self.last_x, atol=0, rtol=0):
                return self.last
            t, y, r, eK = self.runner.simulate(K)
            e = r - y
            m = min(len(t), eK.shape[0])
            J, dJ      = itae_val_grad(t[:m], e[:m], eK[:m,:])
            Ts         = settling_time(t, y, r, tol=0.05)
            OS, dOS    = overshoot_val_grad(t, y, eK, r)
            SSE, dSSE  = sse_val_grad(t, e, eK, W=1.0)
            g  = np.array([OS - 0.10, SSE - SSE_TOL])     # <= 0
            G  = np.vstack([dOS, dSSE])
            
            self.count += 1
            ts_disp = np.nan if np.isinf(Ts) else Ts
            print(f"[sim {self.count:03d}] K=[{K[0]:.6g}, {K[1]:.6g}, {K[2]:.6g}] J={J:.6g}  OS={OS*100:.3f}%  SSE={SSE:.3e}  Ts={ts_disp:.3g}s")
            self.runner.save_plot(self.count, J, OS, SSE)
            
            self.last_x = K.copy()
            self.last   = (J, dJ, g, G)
            return self.last

# --- run it -----------------------------------------------------------------
if __name__ == "__main__":
    runner = SimRunner(MODEL, STOPTIME)
    try:
        if DO_GRAD_CHECK:
            check_gradients(runner, X0, tol=0.05, h=1e-4, rel=1e-1)
        ev = Evaluator(runner)
        f   = lambda K: ev.evaluate(K)[0]
        jf  = (lambda K: ev.evaluate(K)[1]) if USE_ANALYTIC else None
        g   = lambda K: ev.evaluate(K)[2]
        jg  = (lambda K: ev.evaluate(K)[3]) if USE_ANALYTIC else "2-point"
        nlc = NonlinearConstraint(g, [-np.inf, -np.inf], [0.0, 0.0], jac=jg)

        res = minimize(
            f, X0, jac=jf, bounds=BOUNDS, constraints=[nlc],
            method="trust-constr", callback=cb_repeat_header, options={"maxiter": 100, "verbose": 3, "finite_diff_rel_step": 1e-2}
        )
        print("\nOptimized [kp, ki, kd] =", res.x)
        print("Objective (ITAE) =", res.fun)
        print("Status:", res.message)
    finally:
        runner.close()
