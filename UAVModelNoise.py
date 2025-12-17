import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mticker  # type: ignore
import matlab.engine  # type: ignore

# -----------------------------------------
# MATLAB / Simulink setup (mirrors FDOptUAV.py)
# -----------------------------------------
MODEL = "uav_problem"
ENG = matlab.engine.connect_matlab("engine_1")
print(f"Python: connected to MATLAB engine for {MODEL}", flush=True)

ENG.load_system(MODEL, nargout=0)
ENG.set_param(MODEL, "SimulationCommand", "stop", nargout=0)
ENG.set_param(MODEL, "SimulationMode", "accelerator", nargout=0)
ENG.set_param(MODEL, "SolverType", "Variable-step", nargout=0)
ENG.set_param(MODEL, "Solver", "ode45", nargout=0)
ENG.set_param(MODEL, "StopTime", "30", nargout=0)
ENG.set_param(MODEL, "RelTol", "1e-4", nargout=0)
ENG.set_param(MODEL, "AbsTol", "1e-6", nargout=0)
ENG.set_param(MODEL, "MaxStep", "1e-3", nargout=0)

# Design variable names/order
DESIGN_NAMES = ["Kp_h", "Ki_h", "Kp_th", "Ki_th", "Kd_th", "Kp_V", "Ki_V"]


def to_vec(x):
    return np.asarray(x).squeeze()


def tail_value(y, n=20):
    n = min(n, y.size)
    return np.mean(y[-n:]) if y.size else np.nan


def step_metrics(t, y, r, pct_band=0.02):
    if t.size == 0 or y.size == 0:
        return np.inf, np.inf, np.inf

    t_vec = to_vec(t)
    y_vec = to_vec(y)
    if t_vec.size < 2:
        return np.inf, np.inf, np.inf

    n_dense = max(2000, t_vec.size * 5)
    t_dense = np.linspace(t_vec[0], t_vec[-1], n_dense)
    y_dense = np.interp(t_dense, t_vec, y_vec)

    if np.size(r) > 1:
        r_vec = to_vec(r)
        r_dense = np.interp(t_dense, t_vec, r_vec)
        r_target = float(r_dense[-1])
    else:
        r_target = float(r)
        r_dense = np.full_like(t_dense, r_target)

    y_init = float(y_dense[0])
    y_ss = tail_value(y_dense, max(10, y_dense.size // 10))
    delta = y_ss - y_init

    if abs(delta) < 1e-12:
        M = (np.max(y_dense) - y_ss) / (y_ss if y_ss != 0 else 1.0)
        Ts = np.inf
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
        target_edge = hi if delta > 0 else lo
        alpha = (target_edge - e1) / (e2 - e1)
        tcross = tt[j - 1] + alpha * (tt[j] - tt[j - 1])
        return float(tcross)

    Ts = settle_delta(t_dense, y_dense, band_min, band_max)
    M = (np.max(y_dense) - y_ss) / abs(delta)
    SSe = r_target - y_ss
    return Ts, M, SSe


def _apply_offset_formatter(ax, x_baseline, y_baseline):
    """
    Show ticks as baseline plus a small scientific offset (stacked).
    Example: '0.5000\n+1.0e-07'.
    """
    def fmt(base):
        def _f(val, pos):
            delta = val - base
            if abs(delta) < 1e-14:
                return f"{base:.4g}"
            sign = "+" if delta >= 0 else "−"
            return f"{base:.4g}\n{sign}{abs(delta):.1e}"
        return _f

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt(x_baseline)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt(y_baseline)))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(mticker.NullLocator())


def run_once(k):
    """
    Run Simulink and return constraints metrics:
    M_h, |SSe_h|, vmax, |SSe_V|
    """
    try:
        k = np.asarray(k, float)
        ENG.workspace["Kp_h"] = float(k[0])
        ENG.workspace["Ki_h"] = float(k[1])
        ENG.workspace["Kp_th"] = float(k[2])
        ENG.workspace["Ki_th"] = float(k[3])
        ENG.workspace["Kd_th"] = float(k[4])
        ENG.workspace["N_th"] = 1.0
        ENG.workspace["Kp_V"] = float(k[5])
        ENG.workspace["Ki_V"] = float(k[6])

        ENG.sim(MODEL, nargout=0)

        t = to_vec(ENG.workspace["time"])
        h_ref = to_vec(ENG.workspace["h_ref"])
        V_ref = to_vec(ENG.workspace["V_ref"])
        h_out = to_vec(ENG.workspace["h_f_out"])
        V_out = to_vec(ENG.workspace["v_f_out"])
    except Exception:
        return math.nan, math.nan, math.nan, math.nan

    if (t.size == 0 or h_out.size == 0 or V_out.size == 0 or
            np.any(~np.isfinite(t)) or np.any(~np.isfinite(h_out)) or np.any(~np.isfinite(V_out))):
        return math.nan, math.nan, math.nan, math.nan

    # Align references (handle scalar or short vectors)
    def expand_ref(ref, tvec):
        ref = np.asarray(ref).squeeze()
        if ref.size == 0:
            return np.full_like(tvec, np.nan)
        if ref.size == 1:
            return np.full_like(tvec, float(ref))
        if ref.size != tvec.size:
            return np.full_like(tvec, float(ref[-1]))
        return ref

    h_ref = expand_ref(h_ref, t)
    V_ref = expand_ref(V_ref, t)

    Ts_h, M_h, SSe_h = step_metrics(t, h_out, h_ref)
    _, _, SSe_V = step_metrics(t, V_out, V_ref)
    vmax = float(np.max(V_out)) if V_out.size > 0 else math.nan

    return M_h, abs(SSe_h), vmax, abs(SSe_V)


def sweep_variable(var_idx, var_name, base_k, deltas, out_dir):
    csv_path = os.path.join(out_dir, f"uav_noise_{var_name}_microperturb.csv")
    required = {"Kp_h", "Ki_h", "Kp_th", "Ki_th", "Kd_th", "Kp_V", "Ki_V",
                "M_h", "SSe_h_abs", "vmax", "SSe_V_abs"}

    def load_csv(path):
        data = np.genfromtxt(path, delimiter=",", names=True)
        cols = set(data.dtype.names or [])
        if not required.issubset(cols):
            return None
        return {
            "csv": path,
            "var_values": np.asarray(data[var_name], float),
            "Kp_h": np.asarray(data["Kp_h"], float),
            "Ki_h": np.asarray(data["Ki_h"], float),
            "Kp_th": np.asarray(data["Kp_th"], float),
            "Ki_th": np.asarray(data["Ki_th"], float),
            "Kd_th": np.asarray(data["Kd_th"], float),
            "Kp_V": np.asarray(data["Kp_V"], float),
            "Ki_V": np.asarray(data["Ki_V"], float),
            "M_h": np.asarray(data["M_h"], float),
            "SSe_h_abs": np.asarray(data["SSe_h_abs"], float),
            "vmax": np.asarray(data["vmax"], float),
            "SSe_V_abs": np.asarray(data["SSe_V_abs"], float),
        }

    if os.path.isfile(csv_path):
        loaded = load_csv(csv_path)
        if loaded is not None:
            print(f"Loaded existing data from {csv_path}")
            return loaded

    var_vals = []
    cols = {name: [] for name in DESIGN_NAMES}
    m_list, sseh_list, vmax_list, ssev_list = [], [], [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DESIGN_NAMES + ["M_h", "SSe_h_abs", "vmax", "SSe_V_abs"])

    for d in deltas:
        k = np.asarray(base_k, float).copy()
        k[var_idx] += d
        M_h, SSe_h_abs, vmax, SSe_V_abs = run_once(k)

        var_vals.append(k[var_idx])
        for name, val in zip(DESIGN_NAMES, k):
            cols[name].append(val)
        m_list.append(M_h)
        sseh_list.append(SSe_h_abs)
        vmax_list.append(vmax)
        ssev_list.append(SSe_V_abs)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(k) + [M_h, SSe_h_abs, vmax, SSe_V_abs])

    print(f"Saved sweep data to {csv_path}")
    out = {
        "csv": csv_path,
        "var_values": np.asarray(var_vals, float),
        "M_h": np.asarray(m_list, float),
        "SSe_h_abs": np.asarray(sseh_list, float),
        "vmax": np.asarray(vmax_list, float),
        "SSe_V_abs": np.asarray(ssev_list, float),
    }
    for name in DESIGN_NAMES:
        out[name] = np.asarray(cols[name], float)
    return out


def plot_grid(results, base_k, out_dir):
    metrics = [
        ("M_h", "Overshoot M_h"),
        ("SSe_h_abs", "|SSe_h|"),
        ("vmax", "V max"),
        ("SSe_V_abs", "|SSe_V|"),
    ]
    vars_order = DESIGN_NAMES

    fig, axes = plt.subplots(len(metrics), len(vars_order), figsize=(36, 24), squeeze=False)

    for c, var in enumerate(vars_order):
        data = results[var]
        x = data["var_values"]
        for r, (key, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            y = np.asarray(data[key], float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                xm = x[mask]
                ym = y[mask]
                order = np.argsort(xm)
                ax.plot(xm[order], ym[order], marker="o", markersize=3.0, linewidth=0.7, linestyle="-")
                noise_ptp = np.ptp(y[mask])
                y_top = np.nanmax(y[mask])
                ax.annotate(f"noise ~ {noise_ptp:.2e}", xy=(np.mean(x[mask]), y_top),
                            xytext=(0, -10), textcoords="offset points",
                            fontsize=8, color="tab:red", ha="center")
                y_baseline = float(np.nanmean(y[mask]))
            else:
                y_baseline = 0.0
            _apply_offset_formatter(ax, float(base_k[c]), y_baseline)
            ax.grid(False)
            ax.set_box_aspect(1)
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
            if r == len(metrics) - 1:
                ax.set_xlabel(f"{var} (nominal {base_k[c]} + δ{var})")
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(var)

    fig.tight_layout()
    pdf_path = os.path.join(out_dir, "uav_noise_all_params_microperturb.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")


def main():
    base_k = np.array([0.04, 0.01, 2.0, 0.6, 0.25, 0.3, 0.08], float)
    deltas = np.linspace(-1e-6, 1e-6, 50)
    out_dir = "uavNoise"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for idx, name in enumerate(DESIGN_NAMES):
        results[name] = sweep_variable(idx, name, base_k, deltas, out_dir)

    plot_grid(results, base_k, out_dir)


if __name__ == "__main__":
    main()
