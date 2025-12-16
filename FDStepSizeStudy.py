import os
import csv
import numpy as np  # type: ignore
import shutil
import subprocess
import FDOptimization as fdo  # reuse MATLAB engine and helpers

RUN_OBJ_STUDY = False
RUN_CONS_STUDY = True

def simulate(k):
    """Run Simulink for gains k and return Ts (settling time)."""
    try:
        fdo.eng.workspace['kp'] = k[0]
        fdo.eng.workspace['ki'] = k[1]
        fdo.eng.workspace['kd'] = k[2]
        fdo.eng.sim(fdo.model, nargout=0)
        t = fdo.toVec(fdo.eng.workspace['time'])
        y = fdo.toVec(fdo.eng.workspace['y'])
        r = fdo.toVec(fdo.eng.workspace['r'])
    except Exception:
        return np.inf

    if not isinstance(t, np.ndarray) or not isinstance(y, np.ndarray) or t.size == 0 or y.size == 0 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        return np.inf

    yss = fdo.tailValue(y, max(10, y.size // 10))
    ts = fdo.settle(t, y, yss, pct=0.02)
    return float(ts)

def simulate_metrics(k):
    """Return (Ts, overshoot M, steady-state error SSe)."""
    try:
        fdo.eng.workspace['kp'] = k[0]
        fdo.eng.workspace['ki'] = k[1]
        fdo.eng.workspace['kd'] = k[2]
        fdo.eng.sim(fdo.model, nargout=0)
        t = fdo.toVec(fdo.eng.workspace['time'])
        y = fdo.toVec(fdo.eng.workspace['y'])
        r = fdo.toVec(fdo.eng.workspace['r'])
    except Exception:
        return np.inf, np.inf, np.inf

    if not isinstance(t, np.ndarray) or not isinstance(y, np.ndarray) or t.size == 0 or y.size == 0 or np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        return np.inf, np.inf, np.inf

    yss = fdo.tailValue(y, max(10, y.size // 10))
    rss = fdo.tailValue(r, max(10, r.size // 10)) if isinstance(r, np.ndarray) else 0.0
    ts = fdo.settle(t, y, yss, pct=0.02)
    denom = abs(yss) if abs(yss) > 1e-9 else 1.0
    m = float((np.max(y) - yss) / denom)
    sse = float(rss - yss)
    return float(ts), m, sse

def formatH(val):
    s = f"{val:.0e}"
    if "e" in s:
        base, exp = s.split("e")
        exp = exp.lstrip("+")
        exp = exp.lstrip("0") or "0"
        return f"$10^{{{int(exp)}}}$"
    return s

def highlightMatch(val, ref):
    return "--" if not np.isfinite(val) else f"{val:.8g}"

def latexEscape(text):
    return str(text).replace("_", "\\_")

def writeLatex(rows, exactGrad, label, varName, outPath):
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
        f.write("$h$ & $f(k+h)$ & $f(k-h)$ & $\\Delta f$ & $\\partial f/\\partial " + varName + " (fw)$ & $\\partial f/\\partial " + varName + " (cen)$\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            h, fPlus, fMinus, deltaForward, dfForward, dfCentral = row
            hfmt = formatH(h)
            fp = f"{fPlus:.10g}" if np.isfinite(fPlus) else "--"
            fm = f"{fMinus:.10g}" if np.isfinite(fMinus) else "--"
            dfw = highlightMatch(dfForward, np.nan)
            dfc = highlightMatch(dfCentral, np.nan)
            dfText = f"{deltaForward:.10g}" if np.isfinite(deltaForward) else "--"
            f.write(f"{hfmt} & {fp} & {fm} & {dfText} & {dfw} & {dfc}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        safeLabel = latexEscape(label)
        f.write(f"\\caption{{Step size study for {varName}, gains {safeLabel}}}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")

def stepStudy(kStart, hList, outCsvPath, varIndex, varName, outTexPath):
    f0 = simulate(kStart)
    rows = []
    for h in hList:
        kPlus = kStart.copy()
        kMinus = kStart.copy()
        kPlus[varIndex] += h
        kMinus[varIndex] -= h
        fPlus = simulate(kPlus)
        fMinus = simulate(kMinus)

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
        writeLatex(rows, np.nan, fdo.buildLabel(kStart), varName, outTexPath)
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            try:
                subprocess.run(
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error", os.path.basename(outTexPath)],
                    cwd=os.path.dirname(outTexPath),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                print(f"(warn) pdflatex failed for {outTexPath}; please compile manually.")
        else:
            print("(info) pdflatex not found on PATH; skipping PDF generation. Install a LaTeX distribution to compile.")

    print(f"Step size study written to {outCsvPath}")
    for h, fp, fm, df, gfw, gcen in rows:
        print(f"h={h:.0e}  f+= {fp:.6g}  f-= {fm:.6g}  dfFw={gfw:.6g}  dfCen={gcen:.6g}")

def writeLatexSimple(rows, label, varName, outPath):
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
        f.write("$h$ & $f(k+h)$ & $f(k-h)$ & $\\Delta f$ & fw & cen\\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            h, fPlus, fMinus, deltaForward, dfForward, dfCentral = row
            hfmt = formatH(h)
            fp = f"{fPlus:.10g}" if np.isfinite(fPlus) else "--"
            fm = f"{fMinus:.10g}" if np.isfinite(fMinus) else "--"
            dfw = highlightMatch(dfForward, np.nan)
            dfc = highlightMatch(dfCentral, np.nan)
            dfText = f"{deltaForward:.10g}" if np.isfinite(deltaForward) else "--"
            f.write(f"{hfmt} & {fp} & {fm} & {dfText} & {dfw} & {dfc}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        safeLabel = latexEscape(label)
        f.write(f"\\caption{{Step size study for {safeLabel} w.r.t. {varName}}}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")

def stepStudyCons(kStart, hList, outCsvPath, varIndex, varName, consFunc, consLabel, outTexPath):
    f0 = consFunc(kStart)
    rows = []
    for h in hList:
        kPlus = kStart.copy()
        kMinus = kStart.copy()
        kPlus[varIndex] += h
        kMinus[varIndex] -= h
        fPlus = consFunc(kPlus)
        fMinus = consFunc(kMinus)

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
        writeLatexSimple(rows, consLabel, varName, outTexPath)
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            try:
                subprocess.run(
                    [pdflatex, "-interaction=nonstopmode", "-halt-on-error", os.path.basename(outTexPath)],
                    cwd=os.path.dirname(outTexPath),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                print(f"(warn) pdflatex failed for {outTexPath}; please compile manually.")
        else:
            print("(info) pdflatex not found on PATH; skipping PDF generation. Install a LaTeX distribution to compile.")

    print(f"Constraint step size study written to {outCsvPath}")
    for h, fp, fm, df, gfw, gcen in rows:
        print(f"h={h:.0e}  f+= {fp:.6g}  f-= {fm:.6g}  dfFw={gfw:.6g}  dfCen={gcen:.6g}")


if __name__ == "__main__":
    kStart = np.array([-1.0, 1.0, 1.0], dtype=float)
    hList = [10.0 ** (-p) for p in range(1, 23)]  # 1e-1 to 1e-22
    label = fdo.buildLabel(kStart)
    outDir = f"slices_{label}"
    os.makedirs(outDir, exist_ok=True)

    if RUN_OBJ_STUDY:
        for idx, name in enumerate(["kp", "ki", "kd"]):
            outCsv = os.path.join(outDir, f"stepStudy_{label}_{name}.csv")
            outTex = os.path.join(outDir, f"stepStudy_{label}_{name}.tex")
            stepStudy(kStart, hList, outCsv, idx, name, outTex)

    if RUN_CONS_STUDY:
        def cons_m(k):
            _, m, _ = simulate_metrics(k)
            return m
        def cons_sse_pos(k):
            _, _, sse = simulate_metrics(k)
            return sse
        def cons_sse_neg(k):
            _, _, sse = simulate_metrics(k)
            return -sse

        constraints = [
            ("c_M", cons_m),
            ("c_SSe_plus", cons_sse_pos),
            ("c_SSe_minus", cons_sse_neg),
        ]
        for idx, name in enumerate(["kp", "ki", "kd"]):
            for cname, cfunc in constraints:
                outCsv = os.path.join(outDir, f"stepStudy_{label}_{cname}_{name}.csv")
                outTex = os.path.join(outDir, f"stepStudy_{label}_{cname}_{name}.tex")
                stepStudyCons(kStart, hList, outCsv, idx, name, cfunc, cname, outTex)
