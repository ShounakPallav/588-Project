import os
import csv
import numpy as np  # type: ignore
import shutil
import subprocess
import FDOptimization as fdo  # reuse MATLAB engine and helpers


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


def formatH(val):
    s = f"{val:.0e}"
    if "e" in s:
        base, exp = s.split("e")
        exp = exp.lstrip("+")
        exp = exp.lstrip("0") or "0"
        return f"$10^{{{int(exp)}}}$"
    return s


def highlightMatch(val, ref):
    if not np.isfinite(val) or not np.isfinite(ref):
        return "--"
    sval = f"{val:.8g}"
    sref = f"{ref:.8g}"
    prefix = ""
    suffix = sval
    for i in range(min(len(sval), len(sref))):
        if sval[i] == sref[i]:
            prefix += sval[i]
        else:
            suffix = sval[i:]
            break
    if len(prefix) == len(sval):
        suffix = ""
    if prefix:
        return r"\textcolor{blue}{" + prefix + "}" + suffix
    return sval


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
            dfw = highlightMatch(dfForward, exactGrad)
            dfc = highlightMatch(dfCentral, exactGrad)
            dfText = f"{deltaForward:.10g}" if np.isfinite(deltaForward) else "--"
            f.write(f"{hfmt} & {fp} & {fm} & {dfText} & {dfw} & {dfc}\\\\\n")
        if np.isfinite(exactGrad):
            f.write("\\midrule\n")
            f.write(f"Exact &  &  &  & {exactGrad:.8g} & {exactGrad:.8g}\\\\\n")
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

    exactGrad = np.nan
    # choose reference as last finite, non-zero central diff if available
    for _, _, _, _, _, dfCentral in reversed(rows):
        if np.isfinite(dfCentral) and dfCentral != 0:
            exactGrad = dfCentral
            break
    if not np.isfinite(exactGrad):
        for _, _, _, _, _, dfCentral in reversed(rows):
            if np.isfinite(dfCentral):
                exactGrad = dfCentral
                break

    if outTexPath:
        writeLatex(rows, exactGrad, fdo.buildLabel(kStart), varName, outTexPath)
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


if __name__ == "__main__":
    kStart = np.array([-1.0, 1.0, 1.0], dtype=float)
    hList = [10.0 ** (-p) for p in range(1, 23)]  # 1e-1 to 1e-22
    label = fdo.buildLabel(kStart)
    outDir = f"slices_{label}"
    os.makedirs(outDir, exist_ok=True)

    for idx, name in enumerate(["kp", "ki", "kd"]):
        outCsv = os.path.join(outDir, f"stepStudy_{label}_{name}.csv")
        outTex = os.path.join(outDir, f"stepStudy_{label}_{name}.tex")
        stepStudy(kStart, hList, outCsv, idx, name, outTex)
