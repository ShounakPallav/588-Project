import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as mticker  # type: ignore
import matlab.engine  # type: ignore

# -----------------------------------------
# MATLAB / Simulink setup
# -----------------------------------------
MODEL = "ProofofConceptProblem"
ENG = matlab.engine.connect_matlab("engine_1")
print(f"Python: connected to MATLAB engine for {MODEL}", flush=True)

ENG.load_system(MODEL, nargout=0)
ENG.set_param(MODEL, "SimulationCommand", "stop", nargout=0)
ENG.set_param(MODEL, "SimulationMode", "accelerator", nargout=0)
ENG.set_param(MODEL, "SolverType", "Variable-step", nargout=0)
ENG.set_param(MODEL, "Solver", "ode45", nargout=0)
ENG.set_param(MODEL, "StopTime", "25", nargout=0)
ENG.set_param(MODEL, "RelTol", "1e-6", nargout=0)
ENG.set_param(MODEL, "AbsTol", "1e-8", nargout=0)
ENG.set_param(MODEL, "MaxStep", "1e-3", nargout=0)


def to_vec(x):
    return np.asarray(x).squeeze()


def tail_value(y, n=20):
    n = min(n, y.size)
    return float(np.mean(y[-n:])) if y.size else np.nan


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


def run_once(kp, ki, kd):
    """Run Simulink once and return Ts, M, SSe (signed), SSe_abs."""
    try:
        ENG.workspace["kp"] = float(kp)
        ENG.workspace["ki"] = float(ki)
        ENG.workspace["kd"] = float(kd)
        ENG.sim(MODEL, nargout=0)
        t = to_vec(ENG.workspace["time"])
        y = to_vec(ENG.workspace["y"])
        r = to_vec(ENG.workspace["r"])
    except Exception:
        return (math.nan,) * 4

    if t.size == 0 or y.size == 0 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        return (math.nan,) * 4

    yss = tail_value(y, max(10, y.size // 10))
    r_target = tail_value(r, max(10, r.size // 10)) if r.size > 1 else float(r) if r.size else math.nan
    ts = settle(t, y, yss, pct=0.02)
    m = (np.max(y) - yss) / (yss if yss != 0 else 1.0) if np.isfinite(yss) else math.nan
    sse = r_target - yss if np.isfinite(r_target) and np.isfinite(yss) else math.nan
    sse_abs = abs(sse) if math.isfinite(sse) else math.nan
    if not math.isfinite(ts):
        ts = math.nan
    return ts, m, sse, sse_abs


def _apply_offset_formatter(ax, x_baseline, y_baseline):
    """
    Show ticks as baseline plus a small scientific offset (stacked).
    Example label: '0.5000\n+1.0e-07'.
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


def sweep_variable(var_name, base_vals, deltas, out_dir):
    """
    Sweep one design variable (kp/ki/kd) over deltas, log to CSV, return arrays.
    Will reuse existing CSV only if required columns are present; otherwise reruns.
    """
    csv_path = os.path.join(out_dir, f"pid_noise_{var_name}_microperturb.csv")
    required_cols = {"kp", "ki", "kd", "Ts", "M", "SSe", "SSe_abs"}

    def load_csv(path):
        data = np.genfromtxt(path, delimiter=",", names=True)
        cols = set(data.dtype.names or [])
        if not required_cols.issubset(cols):
            return None
        return {
            "csv": path,
            "var_values": np.asarray(data[var_name], float),
            "kp": np.asarray(data["kp"], float),
            "ki": np.asarray(data["ki"], float),
            "kd": np.asarray(data["kd"], float),
            "Ts": np.asarray(data["Ts"], float),
            "M": np.asarray(data["M"], float),
            "SSe": np.asarray(data["SSe"], float),
            "SSe_abs": np.asarray(data["SSe_abs"], float),
        }

    if os.path.isfile(csv_path):
        loaded = load_csv(csv_path)
        if loaded is not None:
            print(f"Loaded existing data from {csv_path}")
            return loaded

    # Rerun sweep
    var_vals, kp_list, ki_list, kd_list = [], [], [], []
    ts_list, m_list, sse_list, sse_abs_list = [], [], [], []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kp", "ki", "kd", "Ts", "M", "SSe", "SSe_abs"])

    for d in deltas:
        kp = base_vals["kp"]
        ki = base_vals["ki"]
        kd = base_vals["kd"]
        if var_name == "kp":
            kp += d
        elif var_name == "ki":
            ki += d
        elif var_name == "kd":
            kd += d

        ts, m, sse, sse_abs = run_once(kp, ki, kd)

        var_vals.append({"kp": kp, "ki": ki, "kd": kd}[var_name])
        kp_list.append(kp)
        ki_list.append(ki)
        kd_list.append(kd)
        ts_list.append(ts)
        m_list.append(m)
        sse_list.append(sse)
        sse_abs_list.append(sse_abs)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([kp, ki, kd, ts, m, sse, sse_abs])

    print(f"Saved sweep data to {csv_path}")
    return {
        "csv": csv_path,
        "var_values": np.asarray(var_vals, float),
        "kp": np.asarray(kp_list, float),
        "ki": np.asarray(ki_list, float),
        "kd": np.asarray(kd_list, float),
        "Ts": np.asarray(ts_list, float),
        "M": np.asarray(m_list, float),
        "SSe": np.asarray(sse_list, float),
        "SSe_abs": np.asarray(sse_abs_list, float),
    }


def plot_grid(results, base_vals, out_dir):
    metrics = [
        ("Ts", "Settling time Ts [s]"),
        ("M", "Overshoot M"),
        ("SSe_abs", "|SSe|"),
    ]
    vars_order = ["kp", "ki", "kd"]

    fig, axes = plt.subplots(len(metrics), len(vars_order), figsize=(12, 10), squeeze=False)

    for c, var in enumerate(vars_order):
        data = results[var]
        x = data["var_values"]
        for r, (key, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            y = np.asarray(data[key], float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                ax.plot(x[mask], y[mask], marker="o", markersize=3, linewidth=0.8)
                noise_ptp = np.ptp(y[mask])
                y_top = np.nanmax(y[mask])
                ax.annotate(f"noise ~ {noise_ptp:.2e}", xy=(np.mean(x[mask]), y_top),
                            xytext=(0, -10), textcoords="offset points",
                            fontsize=8, color="tab:red", ha="center")
                y_baseline = float(np.nanmean(y[mask]))
            else:
                noise_ptp = math.nan
                y_baseline = 0.0
            _apply_offset_formatter(ax, float(base_vals[var]), y_baseline)
            ax.grid(False)
            if r == len(metrics) - 1:
                ax.set_xlabel(f"{var} (nominal {base_vals[var]} + δ{var})")
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(var)

    fig.tight_layout()
    pdf_path = os.path.join(out_dir, "pid_noise_all_params_microperturb.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {pdf_path}")


def main():
    base_vals = {"kp": 0.5, "ki": 0.5, "kd": 0.5}
    deltas = np.linspace(-1e-10, 1e-10, 100)
    out_dir = "simplePIDNoise"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for var in ["kp", "ki", "kd"]:
        results[var] = sweep_variable(var, base_vals, deltas, out_dir)

    plot_grid(results, base_vals, out_dir)


if __name__ == "__main__":
    main()
