import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

# ----------------------------
# configuration
# ----------------------------
CASE_FOLDERS = [
    ("slices_kp-0.5_ki0.5_kd0.5", "kp=-0.5 ki=0.5 kd=0.5"),
    ("slices_kp0.5_ki0.5_kd0.5", "kp=0.5 ki=0.5 kd=0.5"),
    ("slices_kp0.5_ki0.5_kd-0.5", "kp=0.5 ki=0.5 kd=-0.5"),
]

OUT_NAME = "functionvalvsiterationsimplemultistart.pdf"
OUT_DERIV = "dobjectivewrtdesignvariablesfwandcent.pdf"
OUT_UAV = "functionvalvsiterationUAVmultistart.pdf"
OUT_UAV_DERIV = "dobjectivewrtdesignvariablesfwandcentUAV.pdf"
OUT_NM = "functionvalvsiterationsimpleNMmultistart.pdf"

# Toggles
MAKE_TS_PLOT = True
MAKE_DERIV_PLOT = True
MAKE_UAV_TS_PLOT = True
MAKE_UAV_DERIV_PLOT = True
MAKE_TS_NM_PLOT = True

# Derivative CSVs for kp, ki, kd (from step size study)
DERIV_FILES = {
    "kp": os.path.join("slices_kp-1_ki1_kd1", "stepStudy_kp-1_ki1_kd1_kp.csv"),
    "ki": os.path.join("slices_kp-1_ki1_kd1", "stepStudy_kp-1_ki1_kd1_ki.csv"),
    "kd": os.path.join("slices_kp-1_ki1_kd1", "stepStudy_kp-1_ki1_kd1_kd.csv"),
}


def _parts_to_label(parts, limit=3):
    pretty = []
    for p in parts[:limit]:
        i = 0
        while i < len(p) and not p[i].isdigit() and p[i] not in ['-', '.']:
            i += 1
        if i == 0 or i == len(p):
            pretty.append(p)
        else:
            pretty.append(f"{p[:i]}={p[i:]}" )
    return " ".join(pretty)


def find_uav_cases():
    cases = []
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("slicesUAV_"):
            raw = name.replace("slicesUAV_", "")
            label = _parts_to_label(raw.split("_"), limit=3)
            cases.append((name, label))
    cases.sort()
    return cases


def find_uav_deriv_files():
    """Return a mapping for UAV deriv plots; pick the first folder that actually has stepStudy CSVs."""
    keys = ["Kp_h", "Ki_h", "Kp_th", "Ki_th", "Kd_th", "Kp_V", "Ki_V"]
    for name in sorted(os.listdir(".")):
        if not (os.path.isdir(name) and name.startswith("slicesUAV_")):
            continue
        label = name.replace("slicesUAV_", "")
        files = {k: os.path.join(name, f"stepStudy_{label}_{k}.csv") for k in keys}
        if any(os.path.isfile(p) for p in files.values()):
            return files
    return {}


def find_nm_cases():
    cases = []
    for name in os.listdir("."):
        if os.path.isdir(name) and name.startswith("slicesGF_"):
            raw = name.replace("slicesGF_", "")
            label = _parts_to_label(raw.split("_"), limit=3)
            cases.append((name, label))
    cases.sort()
    return cases


UAV_CASE_FOLDERS = find_uav_cases()
UAV_DERIV_FILES = find_uav_deriv_files()
NM_CASE_FOLDERS = find_nm_cases()


# ----------------------------
# helpers
# ----------------------------
def read_ts(folder, ts_col="Ts", iter_col="SLSQPiter", filter_feas=True, iter_filter=None, include_iter0=True):
    iters, ts_vals = [], []
    try:
        files = [f for f in os.listdir(folder) if f.startswith("opt_log_") and f.endswith(".csv")]
    except FileNotFoundError:
        print(f"Missing folder: {folder}")
        return iters, ts_vals
    if not files:
        print(f"No opt_log_*.csv in {folder}")
        return iters, ts_vals
    csv_path = os.path.join(folder, files[0])
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                itv = int(float(row.get(iter_col, "0")))
                tsv = float(row.get(ts_col, "nan"))
                feas_val = row.get("feas", "").lower()
                is_feas = (feas_val in ("true", "1"))
                if filter_feas and not is_feas and not (include_iter0 and itv == 0):
                    continue
                if iter_filter is not None and itv not in iter_filter:
                    continue
                if math.isfinite(tsv):
                    iters.append(itv)
                    ts_vals.append(tsv)
            except Exception:
                continue
    return iters, ts_vals


def plot_ts_multistart(cases, ts_col, title, out_path, legend_loc="best", iter_col="SLSQPiter", line_kwargs=None):
    plt.figure(figsize=(7, 4))
    for folder, label in cases:
        x, y = read_ts(folder, ts_col=ts_col, iter_col=iter_col, filter_feas=False,
                       iter_filter=None, include_iter0=True)
        if x and y:
            xy = sorted(zip(x, y), key=lambda p: p[0])
            xs, ys = zip(*xy)
            if line_kwargs:
                line, = plt.plot(xs, ys, marker="o", label=label, **line_kwargs)
            else:
                line, = plt.plot(xs, ys, marker="o", label=label)
            color = line.get_color()
            if 0 in xs:
                idx0 = xs.index(0)
                plt.plot(xs[idx0], ys[idx0], marker="o", markersize=10,
                         markerfacecolor='none', markeredgecolor=color, markeredgewidth=2)
            plt.plot(xs[-1], ys[-1], marker="x", markersize=15, color=color, mew=2.5)
    plt.xlabel(iter_col)
    plt.ylabel(ts_col)
    plt.title(title)
    plt.grid(False)
    ax = plt.gca()
    if ax.has_data():
        handles, labels = ax.get_legend_handles_labels()
        proxy_init, = ax.plot([], [], marker="o", markersize=10, markerfacecolor='none',
                              markeredgecolor='k', markeredgewidth=2, linestyle='None', label="initial (ring)")
        proxy_final, = ax.plot([], [], marker="x", markersize=15, color='k', mew=2.5,
                               linestyle='None', label="final (cross)")
        handles.extend([proxy_init, proxy_final])
        labels.extend([proxy_init.get_label(), proxy_final.get_label()])
        ax.legend(handles, labels, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_deriv_grid(files_dict, out_path, labels):
    cols = len(labels)
    fig, axes = plt.subplots(2, cols, figsize=(3.2*cols, 6), sharex=True)
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col, key in enumerate(labels):
        path = files_dict.get(key, "")
        h_fw, df_fw = [], []
        h_c, df_c = [], []
        if os.path.isfile(path):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        h = float(row.get("h", "nan"))
                        fw = float(row.get("dfForward", "nan"))
                        ce = float(row.get("dfCentral", "nan"))
                    except Exception:
                        continue
                    if math.isfinite(h):
                        if math.isfinite(fw):
                            h_fw.append(h)
                            df_fw.append(fw)
                        if math.isfinite(ce):
                            h_c.append(h)
                            df_c.append(ce)
        else:
            print(f"Missing derivative CSV: {path}")

        def _plot(ax, xs, ys, title):
            if not xs:
                ax.set_title(title + " (no data)")
                ax.grid(True, which="both", linestyle=":")
                return
            pairs = sorted(zip(xs, ys), key=lambda p: p[0])
            xs_sorted, ys_sorted = zip(*pairs)
            ax.plot(xs_sorted, ys_sorted, marker="o")
            ax.set_title(title)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.grid(True, which="both", linestyle=":")

        _plot(axes[0, col], h_fw, df_fw, f"df/d{key} (fw)")
        _plot(axes[1, col], h_c, df_c, f"df/d{key} (cen)")
        axes[1, col].set_xlabel("h")
        axes[0, col].set_ylabel("df/dk")
        axes[1, col].set_ylabel("df/dk")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    if MAKE_TS_PLOT:
        plot_ts_multistart(CASE_FOLDERS, ts_col="Ts", title="Ts vs Iteration (PID multistart)", out_path=OUT_NAME)

    if MAKE_TS_NM_PLOT and NM_CASE_FOLDERS:
        plot_ts_multistart(NM_CASE_FOLDERS, ts_col="Ts", iter_col="NMiter",
                           title="Ts vs Iteration (Nelder-Mead multistart)",
                           out_path=OUT_NM, legend_loc="best",
                           line_kwargs={"markersize": 2.0, "linewidth": 0.6})

    if MAKE_UAV_TS_PLOT and UAV_CASE_FOLDERS:
        plot_ts_multistart(UAV_CASE_FOLDERS, ts_col="Ts_h", title="Ts_h vs Iteration (UAV multistart)",
                           out_path=OUT_UAV, legend_loc="upper right")

    if MAKE_DERIV_PLOT:
        plot_deriv_grid(DERIV_FILES, OUT_DERIV, ["kp", "ki", "kd"])

    if MAKE_UAV_DERIV_PLOT and UAV_DERIV_FILES:
        plot_deriv_grid(UAV_DERIV_FILES, OUT_UAV_DERIV,
                        ["Kp_h", "Ki_h", "Kp_th", "Ki_th", "Kd_th", "Kp_V", "Ki_V"])
