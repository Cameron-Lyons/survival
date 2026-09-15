"""Regenerate Student-t numerical references using R and mpmath 1.3.0.

Run from the repository root with an interpreter that has mpmath installed:
    python python/tests/fixtures/student_t_numerics_reference.py

mpmath is a generator dependency only. Tests consume the checked-in JSON.
R's dt/pt provide density/CDF references. Quantiles use independent high-precision
references because R qt loses accuracy near the median and in some extreme tails.
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import mpmath as mp

mp.mp.dps = 100
DFS = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 7.0, 30.0, 1e3, 1e8, 1e20]
PROBABILITIES = sorted(
    {
        0.0,
        math.nextafter(0.0, 1.0),
        1e-300,
        1e-100,
        1e-20,
        1e-16,
        0.01,
        0.1,
        0.25,
        0.5 - 1e-9,
        0.5 - 1e-15,
        math.nextafter(0.5, 0.0),
        0.5,
        math.nextafter(0.5, 1.0),
        0.5 + 1e-15,
        0.5 + 1e-9,
        0.75,
        0.9,
        0.99,
        math.nextafter(1.0, 0.0),
        1.0,
    }
)
MAGNITUDES = [
    math.nextafter(0.0, 1.0),
    1e-300,
    1e-155,
    1e-15,
    1e-12,
    1e-9,
    1e-8,
    0.1,
    1.0,
    2.0,
    20.0,
    1e154,
    1e155,
    1e160,
    1e200,
    sys.float_info.max,
]
XS = [-math.inf] + [-x for x in reversed(MAGNITUDES)] + [0.0] + MAGNITUDES + [math.inf]


def r_vector(values):
    # R's decimal parser can round 1e155 differently and even turn maxfloat into
    # Inf. Hex literals preserve exactly the same binary64 inputs in both runtimes.
    return (
        "c("
        + ",".join(
            "Inf" if x == math.inf else "-Inf" if x == -math.inf else float(x).hex() for x in values
        )
        + ")"
    )


def mp_quantile(p, df, r_value):
    if p == 0 or p == 1:
        return (-math.inf if p == 0 else math.inf), "exact endpoint"
    if p == 0.5:
        return 0.0, "exact symmetry"
    nu, probability = mp.mpf(df), mp.mpf(p)
    delta = probability - mp.mpf(".5")
    log_c = mp.loggamma((nu + 1) / 2) - mp.loggamma(nu / 2) - mp.log(nu * mp.pi) / 2
    if abs(delta) <= mp.mpf("1.1e-9"):
        linear = delta / mp.exp(log_c)
        # Invert the density integral through cubic order; the omitted relative
        # term is below 1e-30 for this grid, including df=.1.
        value = linear + (nu + 1) / (6 * nu) * linear**3
        return float(value), "mpmath density-integral inversion through cubic order"
    sign = -1 if p < 0.5 else 1
    tail = probability if p < 0.5 else 1 - probability
    log_tail = mp.log(tail)
    if df >= 1e8:
        # Direct log-erfc inversion avoids forming 1-2p for subnormal p.
        z = mp.mpf(abs(r_value))
        for _ in range(20):
            q = mp.erfc(z / mp.sqrt(2)) / 2
            step = (mp.log(q) - log_tail) * q / (mp.exp(-z * z / 2) / mp.sqrt(2 * mp.pi))
            z += step
            if abs(step) < mp.mpf("1e-85"):
                break
        value = (
            z
            + (z**3 + z) / (4 * nu)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * nu**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * nu**3)
        )
        return sign * float(value), "mpmath normal-tail inversion plus Cornish-Fisher through df^-3"
    if df == 1:
        return sign * float(1 / mp.tan(mp.pi * tail)), "mpmath exact Cauchy quantile"
    if df == 2:
        return sign * float(
            (1 - 2 * tail) / mp.sqrt(2 * tail * (1 - tail))
        ), "mpmath exact df=2 quantile"
    if math.isfinite(r_value) and r_value != 0:
        log_x = mp.log(abs(r_value))
    else:
        log_x = (log_c + (nu - 1) * mp.log(nu) / 2 - log_tail) / nu
    for _ in range(100):
        x = mp.exp(log_x)
        z = nu / (nu + x * x)
        q = mp.betainc(nu / 2, mp.mpf(".5"), 0, z, regularized=True) / 2
        log_pdf = log_c - (nu + 1) * mp.log1p(x * x / nu) / 2
        step = (mp.log(q) - log_tail) * mp.exp(mp.log(q) - log_x - log_pdf)
        log_x += step
        if abs(step) < mp.mpf("1e-85"):
            break
    else:
        raise RuntimeError(f"High-precision inversion did not converge: df={df}, p={p}")
    return sign * float(mp.exp(log_x)), "mpmath log-incomplete-beta inversion"


def encode(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Inf" if value > 0 else "-Inf"
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode(item) for item in value]
    return value


program = f"""
suppressPackageStartupMessages(library(survival))
stopifnot(as.character(packageVersion('survival')) == '3.8.11')
fmt <- function(x) sprintf('%.17g', x)
for (df in {r_vector(DFS)}) {{
  for (p in {r_vector(PROBABILITIES)})
    cat(paste(c('q',fmt(df),fmt(p),fmt(qt(p,df))),collapse='\\t'),'\\n',sep='')
  for (x in {r_vector(XS)})
    cat(paste(c('x',fmt(df),fmt(x),fmt(dt(x,df)),fmt(pt(x,df)),
                fmt(dt(x,df,log=TRUE)),fmt(pt(x,df,log.p=TRUE))),collapse='\\t'),'\\n',sep='')
}}
"""
rscript = shutil.which("Rscript")
if rscript is None:
    raise RuntimeError("Rscript is required to regenerate these fixtures")
result = subprocess.run(  # noqa: S603 - executable and program are locally generated references
    [rscript, "--vanilla", "-e", program], capture_output=True, text=True, check=True
)
cases = {str(df): {"df": df, "points": [], "quantiles": []} for df in DFS}
for line in result.stdout.splitlines():
    fields = line.split("\t")
    kind, df, value = fields[0], float(fields[1]), float(fields[2])
    case = cases[str(df)]
    if kind == "q":
        r_value = float(fields[3])
        expected, source = mp_quantile(value, df, r_value)
        case["quantiles"].append({"p": value, "r": r_value, "expected": expected, "source": source})
    else:
        case["points"].append(
            {
                "x": value,
                "pdf": float(fields[3]),
                "cdf": float(fields[4]),
                "log_pdf": float(fields[5]),
                "log_cdf": float(fields[6]),
            }
        )
for case in cases.values():
    nu = mp.mpf(case["df"])
    case["density_zero_mp"] = float(
        mp.gamma((nu + 1) / 2) / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / 2))
    )
output = {
    "source": "R stats dt/pt/qt with survival 3.8.11 installed; binary-exact inputs",
    "mpmath_version": mp.__version__,
    "mpmath_decimal_precision": mp.mp.dps,
    "notes": [
        "R qt values are retained for audit; independent quantile expectations avoid "
        "R's known median and extreme-tail inaccuracies.",
        "Cornish-Fisher truncation beyond df^-3 is below 1e-22 relative "
        "for df>=1e8 and |z|<40 in this grid.",
        "Positive representable PDF/rare-CDF references must not pass by an absolute "
        "tolerance that permits zero.",
    ],
    "cases": list(cases.values()),
}
path = Path(__file__).with_suffix(".json")
path.write_text(json.dumps(encode(output), indent=2, allow_nan=False) + "\n")
print(f"Wrote {len(cases)} Student-t grids to {path}")
