\name{survivalr}
\alias{survivalr}
\alias{survivalr-package}
\alias{Surv}
\alias{is.Surv}
\alias{tcut}
\alias{neardate}
\alias{survfit}
\alias{survdiff}
\alias{coxph}
\alias{coxph.control}
\alias{survreg}
\alias{survreg.control}
\alias{basehaz}
\alias{concordance}
\alias{cox.zph}
\alias{cox_zph}
\alias{coxph.detail}
\alias{coxph_detail}
\alias{survival_python_config}
\alias{survfit.character}
\alias{survfit.formula}
\alias{survfit.survival_py_coxph}
\alias{coef.survival_py_model}
\alias{vcov.survival_py_model}
\alias{confint.survival_py_model}
\alias{logLik.survival_py_model}
\alias{nobs.survival_py_model}
\alias{df.residual.survival_py_survreg}
\alias{extractAIC.survival_py_model}
\alias{formula.survival_py_model}
\alias{terms.survival_py_model}
\alias{weights.survival_py_model}
\alias{model.matrix.survival_py_model}
\alias{model.frame.survival_py_model}
\alias{fitted.survival_py_model}
\alias{summary.survival_py_model}
\alias{summary.survival_py_survfit}
\alias{summary.survival_py_basehaz}
\alias{summary.survival_py_survdiff}
\alias{summary.survival_py_concordance}
\alias{summary.survival_py_cox_zph}
\alias{summary.survival_py_coxph_detail}
\alias{summary.survival_py_anova}
\alias{predict.survival_py_model}
\alias{residuals.survival_py_model}
\alias{anova.survival_py_model}
\alias{as.data.frame.survival_py_survfit}
\alias{as.data.frame.survival_py_surv}
\alias{as.data.frame.survival_py_basehaz}
\alias{as.data.frame.survival_py_survdiff}
\alias{as.data.frame.survival_py_concordance}
\alias{as.data.frame.survival_py_cox_zph}
\alias{as.data.frame.survival_py_coxph_detail}
\alias{as.data.frame.survival_py_anova}
\alias{print.survival_py_model}
\alias{print.survival_py_object}
\alias{print.survival_py_surv}
\alias{print.survival_py_survfit}
\alias{print.survival_py_basehaz}
\alias{print.survival_py_survdiff}
\alias{print.survival_py_concordance}
\alias{print.survival_py_cox_zph}
\alias{print.survival_py_coxph_detail}
\alias{print.survival_py_anova}
\alias{print.summary.survival_py_model}
\title{R bridge for the Rust-backed Python survival package}
\description{
Experimental R facade for the Rust-backed Python \pkg{survival} package.
The exported functions preserve familiar R survival-analysis entry points while
delegating computation to \code{survival.r_api} through \pkg{reticulate}.
}
\usage{
Surv(time, time2, event, type = NULL, origin = 0)

is.Surv(value)

survfit(formula, ...)

\method{survfit}{formula}(formula, data = NULL, ..., subset = NULL,
  na.action = "fail")

\method{survfit}{character}(formula, data = NULL, ..., subset = NULL,
  na.action = "fail")

\method{survfit}{survival_py_coxph}(formula, newdata = NULL, ..., se.fit = TRUE)

survdiff(formula, data = NULL, ..., group = NULL, subset = NULL,
  na.action = "fail")

coxph(formula, data = NULL, ..., subset = NULL, na.action = "fail")

coxph.control(eps = 1e-09, toler.chol = .Machine$double.eps^0.75,
  iter.max = 20, toler.inf = sqrt(eps), outer.max = 10,
  timefix = TRUE)

survreg(formula, data = NULL, ..., subset = NULL, na.action = "fail")

survreg.control(maxiter = 30, rel.tolerance = 1e-09,
  toler.chol = 1e-10, iter.max, debug = 0, outer.max = 10)

basehaz(fit, ..., centered = TRUE)

concordance(formula, data = NULL, ..., subset = NULL, na.action = "fail")

cox.zph(fit, ...)

cox_zph(fit, ...)

coxph.detail(fit, ...)

coxph_detail(fit, ...)

survival_python_config()

tcut(x, breaks, labels, scale = 1)

neardate(id1, id2, y1, y2, best = c("after", "prior"),
  nomatch = NA_integer_)

\method{coef}{survival_py_model}(object, ...)

\method{vcov}{survival_py_model}(object, ..., complete = TRUE)

\method{confint}{survival_py_model}(object, parm, level = 0.95, ...)

\method{logLik}{survival_py_model}(object, ...)

\method{nobs}{survival_py_model}(object, ...)

\method{df.residual}{survival_py_survreg}(object, ...)

\method{extractAIC}{survival_py_model}(fit, scale = 0, k = 2, ...)

\method{formula}{survival_py_model}(x, ...)

\method{terms}{survival_py_model}(x, ...)

\method{weights}{survival_py_model}(object, ...)

\method{model.matrix}{survival_py_model}(object, ...)

\method{model.frame}{survival_py_model}(formula, ...)

\method{fitted}{survival_py_model}(object, ..., type = NULL, se.fit = FALSE)

\method{summary}{survival_py_model}(object, ...)

\method{summary}{survival_py_survfit}(object, ...)

\method{summary}{survival_py_basehaz}(object, ...)

\method{summary}{survival_py_survdiff}(object, ...)

\method{summary}{survival_py_concordance}(object, ...)

\method{summary}{survival_py_cox_zph}(object, ...)

\method{summary}{survival_py_coxph_detail}(object, ...)

\method{summary}{survival_py_anova}(object, ...)

\method{predict}{survival_py_model}(object, newdata = NULL, ..., type = NULL,
  se.fit = FALSE)

\method{residuals}{survival_py_model}(object, ..., type = "martingale")

\method{anova}{survival_py_model}(object, ..., test = "Chisq")

\method{as.data.frame}{survival_py_survfit}(x, row.names = NULL,
  optional = FALSE, ...)

\method{as.data.frame}{survival_py_surv}(x, row.names = NULL,
  optional = FALSE, ...)

\method{as.data.frame}{survival_py_basehaz}(x, row.names = NULL,
  optional = FALSE, ...)

\method{as.data.frame}{survival_py_survdiff}(x, row.names = NULL,
  optional = FALSE, ...)

\method{as.data.frame}{survival_py_concordance}(x, row.names = NULL,
  optional = FALSE, ...)

\method{as.data.frame}{survival_py_cox_zph}(x, row.names = NULL,
  optional = FALSE, ...)

\method{as.data.frame}{survival_py_coxph_detail}(x, row.names = NULL,
  optional = FALSE, ...)

\method{as.data.frame}{survival_py_anova}(x, row.names = NULL,
  optional = FALSE, ...)

\method{print}{survival_py_model}(x, ...)

\method{print}{survival_py_object}(x, ...)

\method{print}{survival_py_surv}(x, ...)

\method{print}{survival_py_survfit}(x, ...)

\method{print}{survival_py_basehaz}(x, ...)

\method{print}{survival_py_survdiff}(x, ...)

\method{print}{survival_py_concordance}(x, ...)

\method{print}{survival_py_cox_zph}(x, ...)

\method{print}{survival_py_coxph_detail}(x, ...)

\method{print}{survival_py_anova}(x, ...)

\method{print}{summary.survival_py_model}(x, ...)
}
\arguments{
\item{time, time2, event}{Survival response vectors.}
\item{type}{Optional survival response or prediction type.}
\item{origin}{Optional time origin passed to the Python response constructor.}
\item{value}{Object to test.}
\item{breaks}{Strictly increasing cut points for \code{tcut}.}
\item{labels}{Optional interval labels for \code{tcut}.}
\item{id1, id2}{Query and reference IDs for \code{neardate}.}
\item{y1, y2}{Query and reference dates or numeric values for \code{neardate}.}
\item{best}{Nearest-date direction, either \code{"after"} or \code{"prior"}.
The bridge also accepts \code{"closest"} as a Python-backed extension.}
\item{nomatch}{Scalar value used when \code{neardate} cannot find a match.}
\item{formula}{Formula, formula string, or bridged fitted Cox model accepted by
\code{survival.r_api}.}
\item{data, newdata}{Optional data frames or Python-compatible data objects.}
\item{subset}{Optional row subset.}
\item{na.action}{Missing-data policy, either a string or one of the common
\pkg{stats} NA action functions.}
\item{group}{Optional group vector for direct \code{survdiff} calls.}
\item{fit, object}{Bridged model or result object.}
\item{x}{Bridged result object for methods, or a numeric vector for \code{tcut}.}
\item{eps, toler.chol, iter.max, toler.inf, outer.max, timefix}{Control
options passed through \code{coxph.control}.}
\item{maxiter, rel.tolerance, debug}{Control options passed through
\code{survreg.control}.}
\item{row.names, optional}{Standard \code{as.data.frame} arguments.}
\item{centered}{Logical flag for centered baseline hazards.}
\item{complete}{Logical flag requesting the full variance matrix where the
Python fit exposes extra parameters such as estimated survreg scale terms.}
\item{parm}{Optional coefficient names or 1-based coefficient indices.}
\item{level}{Confidence level for coefficient intervals.}
\item{se.fit}{Logical flag requesting standard errors when supported.}
\item{scale, k}{Standard \code{extractAIC} arguments.}
\item{test}{Test name for bridged model analysis of variance.}
\item{...}{Additional arguments passed to the Python API.}
}
\value{
A reticulate-wrapped Python object, an R-converted Python result, or a scalar
value depending on the delegated Python function.
}
\details{
This package is intentionally named \pkg{survivalr} so it can coexist with
CRAN's \pkg{survival} package while the bridge matures. The Python package must
be installed in the Python environment selected by \pkg{reticulate}.
}
\seealso{
\code{\link{survival_python_config}}
}
\keyword{survival}
\keyword{models}
