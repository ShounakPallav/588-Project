import os
import csv
import numpy as np  # type: ignore
import shutil
import subprocess
import FDOptUAV as fdo  # reuse MATLAB engine, model setup, and helpers

# Toggles
RUN_OBJ_STUDY = False  # set True to regenerate objective (Ts_h) step-size tables
RUN_CONS_STUDY = True  # set False if you only want the objective tables

# Design variables order (matching FDOptUAV):
# k[0]=Kp_h, k[1]=Ki_h, k[2]=Kp_th, k[3]=Ki_th, k[4]=Kd_th, k[5]=Kp_V, k[6]=Ki_V
VAR_NAMES = ["Kp_h", "Ki_h", "Kp_th", "Ki_th", "Kd_th", "Kp_V", "Ki_V"]
FMT_SHORT = lambda val: f"{val:.4g}".replace('+', '').replace(' ', '')

# Constraints c >= 0 style (same as FDOptUAV consVals)
CONSTRAINTS = [
    ("c_Mh",  "0.10 - M_h",    lambda Ts_h, M_h, SSe_h, SSe_V, vmax: 0.10 - M_h),
    ("c_SSeh+", "1e-3 - SSe_h", lambda Ts_h, M_h, SSe_h, SSe_V, vmax: 1e-3 - SSe_h),
    ("c_SSeh-", "1e-3 + SSe_h", lambda Ts_h, M_h, SSe_h, SSe_V, vmax: 1e-3 + SSe_h),
    ("c_vmax", "22 - Vmax",     lambda Ts_h, M_h, SSe_h, SSe_V, vmax: 22.0 - vmax),
    ("c_SSeV+", "1e-3 - SSe_V", lambda Ts_h, M_h, SSe_h, SSe_V, vmax: 1e-3 - SSe_V),
    ("c_SSeV-", "1e-3 + SSe_V", lambda Ts_h, M_h, SSe_h, SSe_V, vmax: 1e-3 + SSe_V),
]


def simulate_metrics(k):
    """Run Simulink for gains k and return metrics (Ts_h, M_h, SSe_h, SSe_V, vmax, ok)."""
    try:
        k = np.asarray(k, float)
        eng = fdo.eng
        # push gains
        eng.workspace['Kp_h'] = float(k[0])
        eng.workspace['Ki_h'] = float(k[1])
        eng.workspace['Kp_th'] = float(k[2])
        eng.workspace['Ki_th'] = float(k[3])
        eng.workspace['Kd_th'] = float(k[4])
        eng.workspace['N_th'] = 1.0
        eng.workspace['Kp_V'] = float(k[5])
        eng.workspace['Ki_V'] = float(k[6])

        eng.sim(fdo.model, nargout=0)

        t = fdo.toVec(eng.workspace['time'])
        h_ref = fdo._expand_ref_to_time(t, fdo.toVec(eng.workspace['h_ref']))
        h_out = fdo.toVec(eng.workspace['h_f_out'])
        V_ref = fdo._expand_ref_to_time(t, fdo.toVec(eng.workspace['V_ref']))
        V_out = fdo.toVec(eng.workspace['v_f_out'])

        if (t.size == 0 or h_out.size == 0 or V_out.size == 0 or
                np.any(~np.isfinite(t)) or np.any(~np.isfinite(h_out)) or np.any(~np.isfinite(V_out))):
            return (np.inf, np.inf, np.inf, np.inf, np.inf, False)

        ymax = 1e6
        if np.max(np.abs(h_out)) > ymax or np.max(np.abs(V_out)) > ymax:
            return (np.inf, np.inf, np.inf, np.inf, np.inf, False)

        Ts_h, M_h, SSe_h = fdo.step_metrics(t, h_out, h_ref)
        _, _, SSe_V = fdo.step_metrics(t, V_out, V_ref)
        vmax = float(np.max(V_out)) if V_out.size > 0 else np.inf

        if not (np.isfinite(Ts_h) and np.isfinite(M_h) and np.isfinite(SSe_h) and
                np.isfinite(SSe_V) and np.isfinite(vmax)):
            return (np.inf, np.inf, np.inf, np.inf, np.inf, False)

        return (float(Ts_h), float(M_h), float(SSe_h), float(SSe_V), float(vmax), True)
    except Exception:
        return (np.inf, np.inf, np.inf, np.inf, np.inf, False)


def formatH(val):
    s = f"{val:.0e}"
    if "e" in s:
        base, exp = s.split("e")
        exp = exp.lstrip("+")
        exp = exp.lstrip("0") or "0"
        return f"$10^{{{int(exp)}}}$"
    return s


def highlight(val):
    return "--" if not np.isfinite(val) else f"{val:.10g}"


def latexEscape(text):
    return str(text).replace("_", "\\_")


def writeLatex(rows, label, varName, outPath, descr):
    var_disp = varName.replace("_", "\\_")
    safeLabel = label.replace("_", "\\_")
    with open(outPath, "w", newline="") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[margin=1in]{geometry}\n")
        f.write("\\usepackage[table]{xcolor}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("$h$ & $f(k+h)$ & $f(k-h)$ & $\\Delta f$ & $\\partial f/\\partial "
                + var_disp + " (fw)$ & $\\partial f/\\partial " + var_disp + " (cen)$\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            h, fPlus, fMinus, deltaForward, dfForward, dfCentral = row
            f.write(f"{formatH(h)} & {highlight(fPlus)} & {highlight(fMinus)} & "
                    f"{highlight(deltaForward)} & {highlight(dfForward)} & {highlight(dfCentral)}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Step size study for {descr}, gains \\texttt{{{safeLabel}}}}}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")


def stepStudy(kStart, hList, outCsvPath, varIndex, varName, outTexPath, eval_func, descr):
    f0 = eval_func(kStart)
    rows = []
    outDir = os.path.normpath(os.path.dirname(outCsvPath))
    if outDir:
        os.makedirs(outDir, exist_ok=True)
    if outTexPath:
        texDir = os.path.normpath(os.path.dirname(outTexPath))
        if texDir:
            os.makedirs(texDir, exist_ok=True)
    for h in hList:
        kPlus = kStart.copy()
        kMinus = kStart.copy()
        kPlus[varIndex] += h
        kMinus[varIndex] -= h
        fPlus = eval_func(kPlus)
        fMinus = eval_func(kMinus)

        deltaForward = fPlus - f0
        dfForward = (fPlus - f0) / h if np.isfinite(fPlus) else np.nan
        if np.isfinite(fPlus) and np.isfinite(fMinus):
            dfCentral = (fPlus - fMinus) / (2 * h)
        elif np.isfinite(fPlus):
            dfCentral = (fPlus - f0) / h
        elif np.isfinite(fMinus):
            dfCentral = (f0 - fMinus) / h
        else:
            dfCentral = np.nan

        rows.append((h, fPlus, fMinus, deltaForward, dfForward, dfCentral))

    with open(outCsvPath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['h', 'fPlus', 'fMinus', 'deltaForward', 'dfForward', 'dfCentral'])
        for row in rows:
            writer.writerow(row)

    if outTexPath:
        writeLatex(rows, fdo.buildLabel(kStart), varName, outTexPath, descr)
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            try:
                tex_dir = os.path.dirname(outTexPath) or "."
                tex_file = os.path.basename(outTexPath)
                subprocess.run(
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_file],
                    cwd=tex_dir,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                print(f"(warn) pdflatex failed for {outTexPath}; please compile manually.")
        else:
            print("(info) pdflatex not found on PATH; skipping PDF generation.")

    print(f"Step size study written to {outCsvPath}")
    for h, fp, fm, df, gfw, gcen in rows:
        print(f"h={h:.0e}  f+= {fp:.6g}  f-= {fm:.6g}  dfFw={gfw:.6g}  dfCen={gcen:.6g}")


def build_short_label(k):
    return ("Kph" + FMT_SHORT(k[0]) +
            "_Kih" + FMT_SHORT(k[1]) +
            "_Kpth" + FMT_SHORT(k[2]) +
            "_Kith" + FMT_SHORT(k[3]) +
            "_Kdth" + FMT_SHORT(k[4]) +
            "_KpV" + FMT_SHORT(k[5]) +
            "_KiV" + FMT_SHORT(k[6]))


def eval_objective(k):
    Ts_h, M_h, SSe_h, SSe_V, vmax, ok = simulate_metrics(k)
    return Ts_h if ok else np.inf


def eval_constraint(k, func):
    Ts_h, M_h, SSe_h, SSe_V, vmax, ok = simulate_metrics(k)
    if not ok:
        return np.inf
    return func(Ts_h, M_h, SSe_h, SSe_V, vmax)


if __name__ == "__main__":
    kStart = np.array([0.054469664015827916,
                       0.007635496131252121,
                       1.680203525673334,
                       2.01171264925425,
                       -1.6478568938254035,
                       2.0588185704085777,
                       1.504167233332254], dtype=float)

    hList = [10.0 ** (-p) for p in range(1, 23)]  # 1e-1 to 1e-22
    label = build_short_label(kStart)
    outDir = f"slicesUAV_{label}"
    os.makedirs(outDir, exist_ok=True)

    if RUN_OBJ_STUDY:
        for idx, name in enumerate(VAR_NAMES):
            outCsv = os.path.join(outDir, f"stepStudy_{label}_{name}.csv")
            outTex = os.path.join(outDir, f"stepStudy_{label}_{name}.tex")
            stepStudy(kStart, hList, outCsv, idx, name, outTex, eval_objective, descr=name)

    if RUN_CONS_STUDY:
        for c_name, c_desc, c_func in CONSTRAINTS:
            for idx, vname in enumerate(VAR_NAMES):
                outCsv = os.path.join(outDir, f"stepStudy_{label}_{c_name}_{vname}.csv")
                outTex = os.path.join(outDir, f"stepStudy_{label}_{c_name}_{vname}.tex")
                stepStudy(kStart, hList, outCsv, idx, vname, outTex,
                          lambda k, f=c_func: eval_constraint(k, f),
                          descr=f"{c_desc} w.r.t. {vname}")
