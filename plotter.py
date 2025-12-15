import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt  # type: ignore

# Folders and labels to plot (Ts vs SLSQPiter)
CASE_FOLDERS = [
    ("slices_kp-0.5_ki0.5_kd0.5", "kp=-0.5 ki=0.5 kd=0.5"),
    ("slices_kp0.5_ki0.5_kd0.5", "kp=0.5 ki=0.5 kd=0.5"),
    ("slices_kp0.5_ki0.5_kd-0.5", "kp=0.5 ki=0.5 kd=-0.5"),
]

OUT_NAME = "functionvalvsiterationsimplemultistart.pdf"
OUT_DERIV = "dobjectivewrtdesignvariablesfwandcent.pdf"

# Toggles
MAKE_TS_PLOT = False
MAKE_DERIV_PLOT = True

# Derivative CSVs for kp, ki, kd (from step size study)
DERIV_FILES = {
    "kp": os.path.join("slices_kp-1_ki1_kd1", "stepStudy_kp-1_ki1_kd1_kp.csv"),
    "ki": os.path.join("slices_kp-1_ki1_kd1", "stepStudy_kp-1_ki1_kd1_ki.csv"),
    "kd": os.path.join("slices_kp-1_ki1_kd1", "stepStudy_kp-1_ki1_kd1_kd.csv"),
}


def read_ts(folder):
    # find first opt_log_*.csv in the folder
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
                itv = int(row.get("SLSQPiter", "0"))
                tsv = float(row.get("Ts", "nan"))
                feas_val = row.get("feas", "").lower()
                if feas_val and feas_val not in ("true", "1"):
                    continue
                if math.isfinite(tsv):
                    iters.append(itv)
                    ts_vals.append(tsv)
            except Exception:
                continue
    return iters, ts_vals


def main():
    if MAKE_TS_PLOT:
        plt.figure(figsize=(7, 4))
        for folder, label in CASE_FOLDERS:
            x, y = read_ts(folder)
            if x and y:
                # sort by iteration to keep lines monotone
                xy = sorted(zip(x, y), key=lambda p: p[0])
                xs, ys = zip(*xy)
                plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel("SLSQPiter")
        plt.ylabel("Ts (settling time)")
        plt.title("Ts vs Iteration (PID multistart)")
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_NAME)
        plt.close()
        print(f"Saved {OUT_NAME}")

    if MAKE_DERIV_PLOT:
        # Derivative plot: 2x3 grid, top row forward, bottom row central, columns kp/ki/kd
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
        for col, key in enumerate(["kp", "ki", "kd"]):
            path = DERIV_FILES.get(key, "")
            h_vals_fw, df_fw = [], []
            h_vals_c, df_c = [], []
            if not os.path.isfile(path):
                print(f"Missing derivative CSV: {path}")
            else:
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            h = float(row.get("h", "nan"))
                            fw = float(row.get("dfForward", "nan"))
                            ce = float(row.get("dfCentral", "nan"))
                            if math.isfinite(h) and math.isfinite(fw):
                                h_vals_fw.append(h)
                                df_fw.append(fw)
                            if math.isfinite(h) and math.isfinite(ce):
                                h_vals_c.append(h)
                                df_c.append(ce)
                        except Exception:
                            continue
            # plot forward: sort, drop zero derivatives, center by median
            if h_vals_fw:
                xy_fw = sorted(zip(h_vals_fw, df_fw), key=lambda p: p[0])
                xs_fw, ys_fw = zip(*xy_fw)
                xs_fw = [x for x, y in zip(xs_fw, ys_fw) if y != 0]
                ys_fw = [y for y in ys_fw if y != 0]
                if ys_fw:
                    med_fw = np.median(ys_fw)
                    ys_fw_center = [y - med_fw for y in ys_fw]
                    axes[0, col].plot(xs_fw, ys_fw_center, marker="o")
            # plot central
            if h_vals_c:
                xy_c = sorted(zip(h_vals_c, df_c), key=lambda p: p[0])
                xs_c, ys_c = zip(*xy_c)
                xs_c = [x for x, y in zip(xs_c, ys_c) if y != 0]
                ys_c = [y for y in ys_c if y != 0]
                if ys_c:
                    med_c = np.median(ys_c)
                    ys_c_center = [y - med_c for y in ys_c]
                    axes[1, col].plot(xs_c, ys_c_center, marker="o")

            axes[0, col].set_title(f"df/d{key} (fw)")
            axes[1, col].set_title(f"df/d{key} (cen)")
            axes[0, col].set_xscale("log")
            axes[1, col].set_xscale("log")
            axes[1, col].set_xlabel("h")
            axes[0, col].set_ylabel("df/dk")
            axes[1, col].set_ylabel("df/dk")
            for ax in axes[:, col]:
                ax.set_yscale("symlog", linthresh=1e-5)
                ax.grid(True, which="both", linestyle=":")

        fig.tight_layout()
        fig.savefig(OUT_DERIV)
        plt.close(fig)
        print(f"Saved {OUT_DERIV}")


if __name__ == "__main__":
    main()
