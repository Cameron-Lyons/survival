# Student-t tails through a transformed normal integral

[`StudentT::normal_integral_tail`](../src/internal/student_t.rs) evaluates the
positive Student-t tail for `1000 <= df < 100000` and `1 < x < 9`.
For `df >= 100000`, `normal_limit_tail` uses an expansion in inverse degrees
of freedom. The center and more distant tails also use other formulas.
The transformation below
avoids an incomplete-beta argument close to one and uses eight correction terms.
Its analytic relative truncation error is below `4.744e-16` on this domain.
That bound does **not** include floating-point rounding.

## Transformation and coefficients

Let ν be the degrees of freedom, φ the standard normal density, Q its upper
tail, and

$$
c_\nu = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}.
$$

In the Student density integral, substitute

$$
v^2=\nu\log(1+t^2/\nu),\qquad
w=\sqrt{\nu\log(1+x^2/\nu)}.
$$

Combining the density with the Jacobian gives the exact identity

$$
\Pr(T_\nu>x)=c_\nu\sqrt{2\pi}
\int_w^\infty \phi(v)g(v^2/\nu)\,dv,
\qquad
g(s)=\sqrt{\frac{s}{1-e^{-s}}}.
$$

The Taylor coefficients of g through degree eight are

| k | coefficient aₖ |
|---:|---:|
| 0 | 1 |
| 1 | 1/4 |
| 2 | 1/96 |
| 3 | −1/384 |
| 4 | −1/10240 |
| 5 | 19/368640 |
| 6 | 79/61931520 |
| 7 | −55/49545216 |
| 8 | −2339/118908518400 |

They follow by expanding `(1-exp(-s))/s`, taking its reciprocal, and then its
square root with constant term one. Define the normal moments

$$
M_j(w)=\int_w^\infty v^j\phi(v)\,dv.
$$

Integration by parts yields

$$
M_0(w)=Q(w),\qquad
M_{2k}(w)=w^{2k-1}\phi(w)+(2k-1)M_{2k-2}(w).
$$

The implementation therefore evaluates

$$
c_\nu\sqrt{2\pi}
\left[Q(w)+\sum_{k=1}^{8}a_k\nu^{-k}M_{2k}(w)\right].
$$

It obtains Q directly from `erfc`. For `r=x²/ν < 1e-8`, it evaluates the lower
limit as `w=x*(1-r/4+13*r²/96)`; this avoids multiplying ν by a quantized
subnormal `log1p(r)`. The normalization factor is evaluated from the cached
log density at zero.

## Bounding the infinite integral

Although `w²/ν < .081`, the integration variable `s=v²/ν` is unbounded. A local
Taylor estimate at the lower limit is therefore insufficient. Split the
integral at `s=.4`, corresponding to `v=sqrt(.4*ν) >= 20`.

For the first part, write

$$
g(s)=e^{s/4}\sqrt{\frac{s/2}{\sinh(s/2)}}.
$$

This branch is analytic in `|s| < 2π`. The product for sinh gives, on `|s|=R`
with `R < 2π`,

$$
\left|\frac{\sinh(s/2)}{s/2}\right|
\geq \prod_{j=1}^{\infty}
\left(1-\frac{R^2}{4\pi^2j^2}\right)
=\frac{\sin(R/2)}{R/2}.
$$

Taking `R=5.5` consequently gives

$$
|g(s)|\leq A=e^{5.5/4}\sqrt{\frac{2.75}{\sin(2.75)}}
<10.616525.
$$

Cauchy's coefficient bound controls the entire remainder after degree eight:

$$
\left|g(s)-\sum_{k=0}^{8}a_ks^k\right|
\leq\frac{A(s/5.5)^9}{1-.4/5.5},\qquad 0\leq s\leq .4.
$$

Because `g(s) >= 1`, the exact transformed integral is at least Q(w). Also,
`M18(w)/Q(w)` increases with w. Using `w < 9` and `ν >= 1000`, the relative
contribution of this remainder is bounded by

$$
\frac{A}{5.5^9(1-.4/5.5)}
\frac{M_{18}(9)}{1000^9Q(9)}
<4.744\times10^{-16}.
$$

For the second part, `g(s) <= sqrt(1+s) <= 1+s/2`. Bound the degree-eight
polynomial by its absolute coefficients and integrate beyond `v=20`, using
`ν=1000` and the same normal-moment recurrence. This contributes less than
`5.7e-70` relative to Q(9). The combined bound remains below `4.744e-16`.
The normalization factor cancels when forming relative error.

## Numerical validation

The bound concerns truncating g in exact arithmetic. Evaluation of w, the
normal tail, the normalization, and the moment sum introduces rounding. It
does not provide a uniform error bound for the full CDF or inverse CDF.

The checked-in [R reference fixture](../python/tests/fixtures/student_t_normal_reference.json)
contains 210 CDF values and 98 quantiles, including both sides of the
`df=1000`, `df=100000`, `x=1`, and `x=9` boundaries. Native and Python tests compare relative CDF and
quantile errors against `1e-13`, check logarithmic tails, and check ordering.
The fixture includes a control below the new branch's domain. Its
[generator](../scripts/generate_student_t_normal_reference.R) records the R
version and can be rerun with R 4.6.1:

```sh
Rscript scripts/generate_student_t_normal_reference.R
```

An additional 4,048-point R comparison included 2,944 points with
`1000 <= df < 100000` and observed maximum relative CDF error below
`3.54e-14` in that region on macOS arm64. The 920 points with `df >= 100000`
matched the existing large-df implementation exactly and had maximum relative
error below `1.83e-14`. These are sampled floating-point results, not exhaustive
bounds or replacements for the analytic truncation argument.

## Performance

For the sampled `df=1000` through `30000` cases, CDF calls are about 3.1–3.4 times
faster and quantile calls about 8.0–13.5 times faster. The `df=100000` and `1e6`
CDF controls retain about 9% measured overhead, while their quantile intervals
include no change. Their results, as well as both low-degree controls, are
bit-for-bit identical to the parent across all 8,192 benchmark outputs.
Compiler output shows that adding the new path changes whether the shared
`erfc` helper is inlined in the existing high-degree expansion; this likely
contributes to the overhead. The table includes every case and run range.

Measured against PR #625 (`7194c31e502763f0250655ef0fcf7edb7ec1d951`) on macOS arm64 with Rust 1.94.0. Both revisions use identical benchmark source and the Cargo bench profile (optimization level 3, LTO, one codegen unit). Each call evaluates 1,024 values through `survreg_distribution`, including input clones, cached distribution construction, allocations, and output destruction. CDF inputs span `[-8.5, 8.5]`; quantile probabilities span `[.001, .999]`; location is zero and scale is one.

Five alternating-order pairs use the Divan OS timer, one Rayon thread, a minimum of 50 samples and .25 seconds per case, and one iteration per sample. Times below are the median of the five run medians, with their full range in brackets. Ratios use the geometric mean of paired median ratios and a 95% Student-t interval on log ratios (four degrees of freedom). These describe repeated runs on this host; they are not cross-machine guarantees.

### CDF

| df | Before µs [range] | After µs [range] | After/before [95% interval] |
|---:|---:|---:|---:|
| 4.5 | 53.04 [52.7, 59.74] | 52.83 [52.66, 59.7] | 0.9748 [0.9110, 1.0430] |
| 999 | 167.7 [166.9, 187.5] | 167.5 [167.2, 187.1] | 0.9775 [0.9185, 1.0403] |
| 1000 | 170.3 [168.7, 190.1] | 50.33 [50.29, 56.99] | 0.2909 [0.2726, 0.3104] |
| 3000 | 165.6 [164.7, 186.1] | 50.24 [50.2, 56.95] | 0.2976 [0.2781, 0.3185] |
| 9999 | 161.6 [160.4, 181.2] | 50.12 [50.04, 56.79] | 0.3049 [0.2849, 0.3265] |
| 10000 | 159.8 [159.4, 179.5] | 50.16 [50.04, 56.74] | 0.3127 [0.2855, 0.3425] |
| 30000 | 153.7 [153.4, 172.5] | 50.16 [49.99, 56.7] | 0.3214 [0.3062, 0.3374] |
| 100000 | 36.74 [33.33, 37.74] | 39.37 [37.08, 42.04] | 1.0949 [1.0345, 1.1589] |
| 1e+06 | 36.58 [33.24, 37.7] | 37.29 [37.08, 41.95] | 1.0942 [1.0417, 1.1493] |

### Quantiles

| df | Before µs [range] | After µs [range] | After/before [95% interval] |
|---:|---:|---:|---:|
| 4.5 | 538.4 [535.9, 602.4] | 537.4 [533.8, 601.7] | 0.9973 [0.9952, 0.9993] |
| 999 | 2707 [2698, 3030] | 2706 [2698, 3032] | 0.9996 [0.9985, 1.0008] |
| 1000 | 2192 [2188, 2460] | 275.5 [273, 309.5] | 0.1253 [0.1244, 0.1262] |
| 3000 | 2614 [2608, 2931] | 257.8 [256.7, 287.7] | 0.0984 [0.0980, 0.0988] |
| 9999 | 2618 [2613, 2936] | 235.3 [231, 262.6] | 0.0892 [0.0882, 0.0902] |
| 10000 | 3125 [3120, 3505] | 234.1 [228.9, 260.8] | 0.0743 [0.0735, 0.0751] |
| 30000 | 2181 [2172, 2445] | 234.2 [232.3, 260.9] | 0.1070 [0.1065, 0.1074] |
| 100000 | 208.4 [207.2, 236.6] | 208.1 [206.7, 235.2] | 0.9973 [0.9935, 1.0012] |
| 1e+06 | 207.8 [206.5, 235.2] | 206.9 [205.9, 234] | 0.9966 [0.9920, 1.0012] |

Run the current cases with:

```sh
RAYON_NUM_THREADS=1 cargo bench --bench survival_benchmarks -- student_t_normal_limit_bench --bench --timer os --sample-count 50 --sample-size 1 --min-time .25
```

For a matched baseline, apply only the same benchmark module to the parent revision and build it with the same toolchain and profile in a separate workspace. Run saved binaries only after builds and correctness checks finish.
