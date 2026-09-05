"""Differential fixtures from R survival 3.8.11 (generated 2026-09-04).

The embedded R script generates the reference values by calling the package's
public residuals generic. Tests use its recorded output and do not require R.
Matrix columns are g, dg, ddg, ds, dds, dsg in R's diagnostic conventions,
including the interval-censoring scale columns.
"""

import math

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

# Run this script with Rscript to regenerate the JSON used below. Constructing
# the residuals object fixes eta and sigma independently of optimizer behavior.
_R_REFERENCE_GENERATOR = r"""
library(survival)
stopifnot(packageVersion("survival") == "3.8.11")
reference_matrix <- function(dist, time, time2, status, eta, scale, parms=NULL) {
    object <- structure(list(
        terms=terms(Surv(time, time2, status, type="interval") ~ 1),
        coefficients=c("(Intercept)"=0), var=diag(2),
        x=matrix(1, length(time), 1),
        y=Surv(time, time2, status, type="interval"),
        linear.predictors=eta, scale=scale, dist=dist, parms=parms
    ), class="survreg")
    unname(residuals(object, type="matrix"))
}
moderate <- list(
    time=c(0.4, 3.1, 2.75, 0.25, 0.75, 2.5),
    time2=c(1, 1, 1, 1, 1.4, 4),
    status=c(1L, 1L, 0L, 2L, 3L, 3L),
    eta=c(0.2, 0.1, 0.65, 1.5, 0.8, -0.15)
)
matrices <- list()
for (dist in c("extreme", "gaussian", "logistic", "weibull", "exponential",
               "rayleigh", "lognormal", "loglogistic", "t")) {
    scale <- switch(dist, exponential=1, rayleigh=0.5, 1.3)
    parms <- if (dist == "t") 5 else NULL
    matrices[[dist]] <- list(
        scale=scale,
        matrix=do.call(reference_matrix,
                       c(list(dist=dist, scale=scale, parms=parms), moderate))
    )
}
tails <- list()
for (dist in c("gaussian", "lognormal", "logistic", "loglogistic", "extreme", "weibull", "t")) {
    bound <- switch(dist, gaussian=9, lognormal=9, extreme=5, weibull=5, 40)
    left <- switch(dist, extreme=-5, weibull=-5, -bound)
    z <- c(bound, left, bound, left - 0.5)
    z2 <- c(bound, left, bound + 0.5, left)
    transformed <- dist %in% c("lognormal", "loglogistic", "weibull")
    input <- list(time=if (transformed) exp(z) else 100 + z,
                  time2=if (transformed) exp(z2) else 100 + z2,
                  status=c(0L, 2L, 3L, 3L),
                  eta=rep(if (transformed) 0 else 100, 4))
    parms <- if (dist == "t") 5 else NULL
    tails[[dist]] <- c(input, list(
        matrix=do.call(reference_matrix,
                       c(list(dist=dist, scale=1, parms=parms), input))
    ))
}
fit_data <- data.frame(
    time=c(1.1, 2.3, 3.2, 1.8, 4.5, 2.7, 5.2, 3.9, 6.3, 4.1, 7.2, 5.6),
    status=c(1L, 0L, 1L, 1L, 0L, 1L, 1L, 0L, 1L, 1L, 0L, 1L),
    x=c(-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2)
)
fits <- list()
for (dist in c("gaussian", "weibull", "logistic", "lognormal")) {
    fit <- survreg(Surv(time, status) ~ x, data=fit_data, dist=dist,
                   control=survreg.control(maxiter=100, rel.tolerance=1e-12))
    fits[[dist]] <- list(coefficients=unname(coef(fit)), scale=unname(fit$scale),
                        working=unname(residuals(fit, type="working")))
}
cat(jsonlite::toJSON(list(moderate=moderate, matrices=matrices, tails=tails,
                         fit_data=as.list(fit_data), fits=fits),
                    auto_unbox=TRUE, digits=NA, pretty=TRUE))
"""

_REFERENCE = {
    "moderate": {
        "time": [0.4, 3.1, 2.75, 0.25, 0.75, 2.5],
        "time2": [1, 1, 1, 1, 1.4, 4],
        "status": [1, 1, 0, 2, 3, 3],
        "eta": [0.2, 0.1, 0.65, 1.5, 0.8, -0.15],
    },
    "matrices": {
        "extreme": {
            "scale": 1.3,
            "matrix": [
                [
                    -1.27482955106727,
                    0.1279318772661,
                    -0.690125112689899,
                    -0.97441362454678,
                    -0.053191379960816,
                    -0.26595689980408,
                ],
                [
                    -8.00587474427604,
                    6.96246368269297,
                    -5.94745727071056,
                    19.8873910480789,
                    -74.4145064844738,
                    -24.8048354948246,
                ],
                [
                    -5.02982209844912,
                    3.86909392188394,
                    -2.97622609375688,
                    8.12509723595627,
                    -21.2502543094241,
                    -10.1191687187734,
                ],
                [
                    -1.14660814154976,
                    -0.631536931100905,
                    -0.0987637693044435,
                    0.789421163876131,
                    -0.943739553414324,
                    0.754991642731459,
                ],
                [
                    -1.72942017627012,
                    0.186251055647005,
                    -0.716139664587,
                    0.924431303199094,
                    -2.00778579569016,
                    -0.850862048028717,
                ],
                [
                    -7.67878664689644,
                    5.90675817329485,
                    -4.54366963754811,
                    -15.652907534791,
                    -16.2551600007304,
                    191.093535366261,
                ],
            ],
        },
        "gaussian": {
            "scale": 1.3,
            "matrix": [
                [
                    -1.19313711719879,
                    0.118343195266272,
                    -0.591715976331361,
                    -0.976331360946746,
                    -0.0473372781065089,
                    -0.236686390532544,
                ],
                [
                    -3.84402469116329,
                    1.77514792899408,
                    -0.591715976331361,
                    4.32544378698225,
                    -10.6508875739645,
                    -3.55029585798816,
                ],
                [
                    -2.93532009990303,
                    1.56718900890766,
                    -0.508686763187659,
                    3.29109691870608,
                    -5.53440554436366,
                    -2.63543121160174,
                ],
                [
                    -1.78295386324124,
                    -1.14955275617863,
                    -0.47121062490458,
                    1.43694094522329,
                    -2.1732075466367,
                    1.73856603730936,
                ],
                [
                    -1.64437110920298,
                    0.159360636956006,
                    -0.579497711455973,
                    0.936432209047963,
                    -1.99602048468075,
                    -0.701096374769624,
                ],
                [
                    -3.90971595676108,
                    1.82321736977769,
                    -0.548450694523126,
                    -4.69088379216174,
                    -0.169833893619714,
                    15.8424963339846,
                ],
            ],
        },
        "logistic": {
            "scale": 1.3,
            "matrix": [
                [
                    -1.65456995907874,
                    0.0590551639425241,
                    -0.294114231971541,
                    -0.988188967211495,
                    -0.0235756020673665,
                    -0.117878010336832,
                ],
                [
                    -2.75975049995052,
                    0.630018633860168,
                    -0.0973962486601644,
                    0.890055901580503,
                    -2.76662213952198,
                    -0.922207379840661,
                ],
                [
                    -1.79671750755912,
                    0.641659713788086,
                    -0.0818572069229963,
                    1.34748539895498,
                    -1.70847568148539,
                    -0.813559848326378,
                ],
                [
                    -1.28529033116856,
                    -0.55648440384357,
                    -0.118390034312382,
                    0.695605504804463,
                    -0.880589933417559,
                    0.704471946734047,
                ],
                [
                    -2.09562557612143,
                    0.0798191376925389,
                    -0.288196894518159,
                    0.968048458921763,
                    -1.9983588009208,
                    -0.357377156274524,
                ],
                [
                    -2.58008907835093,
                    0.650246379138462,
                    -0.0827493862694615,
                    -1.14598913219758,
                    -0.559926570139007,
                    0.734855023332563,
                ],
            ],
        },
        "weibull": {
            "scale": 1.3,
            "matrix": [
                [
                    -1.54476827374337,
                    -0.443293206991427,
                    -0.250721201722571,
                    -0.505155901532699,
                    -0.807269041055446,
                    0.723170960758683,
                ],
                [
                    -1.67984832639136,
                    0.93143869348647,
                    -1.30820727901326,
                    -0.0393121648135429,
                    -2.35234606938289,
                    -2.28072644332877,
                ],
                [
                    -1.32069029074767,
                    1.01591560826744,
                    -0.781473544821108,
                    0.367356010137904,
                    -0.469537754889008,
                    -1.29849715452737,
                ],
                [
                    -2.27402746451998,
                    -0.728223151732944,
                    -0.0309632733157331,
                    2.10186637648375,
                    -2.35981196698366,
                    0.817592272905958,
                ],
                [
                    -1.88440128660621,
                    -0.337936195201514,
                    -0.328264046217424,
                    0.731035405863976,
                    -1.91185885268832,
                    1.21715422347861,
                ],
                [
                    -2.73607246923448,
                    1.29641182708007,
                    -1.54298508573833,
                    -0.684441593001632,
                    -2.78532161730665,
                    -1.79558207228434,
                ],
            ],
        },
        "exponential": {
            "scale": 1,
            "matrix": [
                [
                    -1.44378303310535,
                    -0.672507698768807,
                    -0.327492301231193,
                    -0.249285888750364,
                    -1.15880390464266,
                    1.03808431939333,
                ],
                [
                    -1.77359388442037,
                    1.80499599591147,
                    -2.80499599591147,
                    0.861676681416077,
                    -4.84560425713066,
                    -4.69807478881865,
                ],
                [
                    -1.43562588609279,
                    1.43562588609279,
                    -1.43562588609279,
                    0.51912362924038,
                    -0.706839206847542,
                    -1.95474951533317,
                ],
                [
                    -2.91405598067643,
                    -0.972368024182101,
                    -0.0273727085094435,
                    2.80654034513009,
                    -3.03457403485186,
                    1.05137371840149,
                ],
                [
                    -1.71025985564678,
                    -0.52406971797621,
                    -0.46885205775027,
                    0.587538242155723,
                    -1.8285571422663,
                    1.47259630459492,
                ],
                [
                    -3.09700373547701,
                    2.53481340264314,
                    -3.3159658988594,
                    -2.23939663038169,
                    -3.63960124220282,
                    4.76670746322086,
                ],
            ],
        },
        "rayleigh": {
            "scale": 0.5,
            "matrix": [
                [
                    -1.64668549055407,
                    -1.7854975852686,
                    -0.42900482946281,
                    0.993134406219017,
                    -2.52771946840799,
                    2.26439170032718,
                ],
                [
                    -5.11205113353726,
                    13.7360050741588,
                    -31.4720101483176,
                    13.1673446369399,
                    -47.6469642448168,
                    -46.1963027940029,
                ],
                [
                    -2.06102168481972,
                    4.12204336963944,
                    -8.24408673927888,
                    1.49053464043985,
                    -2.56849201018267,
                    -7.10311265051915,
                ],
                [
                    -5.77414416468356,
                    -1.99688992199767,
                    -0.00621692846421285,
                    5.763612121639,
                    -5.81540345737553,
                    2.01483380756741,
                ],
                [
                    -1.5166422200774,
                    -1.50396660216242,
                    -0.965635809331009,
                    -0.120468398681807,
                    -1.20139898771925,
                    1.53352756927775,
                ],
                [
                    -8.43661947131499,
                    16.8731844515706,
                    -33.747701946655,
                    -17.9916811374745,
                    -20.3814258235076,
                    420.499341264421,
                ],
            ],
        },
        "lognormal": {
            "scale": 1.3,
            "matrix": [
                [
                    -1.5499729154438,
                    -0.660527060280565,
                    -0.591715976331361,
                    -0.262659764456724,
                    -1.47468047108655,
                    1.32105412056113,
                ],
                [
                    -1.49603366027225,
                    0.610297107391184,
                    -0.591715976331361,
                    -0.370538274799822,
                    -1.25892345040036,
                    -1.22059421478237,
                ],
                [
                    -0.940463395304387,
                    0.756143293951229,
                    -0.409964453500827,
                    0.273422104452333,
                    -0.327027096485988,
                    -0.904386814092898,
                ],
                [
                    -4.32740963372848,
                    -1.97665126426136,
                    -0.531293772021965,
                    5.70519739793806,
                    -10.1312438419197,
                    3.51012148254647,
                ],
                [
                    -1.83681520854581,
                    -0.450201181089082,
                    -0.580481979366469,
                    0.638483480292066,
                    -1.97309040601574,
                    1.67410886073211,
                ],
                [
                    -2.43731934313821,
                    0.761662792255887,
                    -0.585340117914439,
                    0.00880474588451337,
                    -1.95709543871745,
                    -1.74577109680478,
                ],
            ],
        },
        "loglogistic": {
            "scale": 1.3,
            "matrix": [
                [
                    -1.82759412941438,
                    -0.3113628880695,
                    -0.24738456413219,
                    -0.652428493798447,
                    -0.655838648011583,
                    0.587515984218992,
                ],
                [
                    -1.80206206959439,
                    0.290089167738085,
                    -0.253782125546193,
                    -0.700801419874244,
                    -0.569169547551211,
                    -0.551840587885128,
                ],
                [
                    -0.84186468053761,
                    0.437764406701653,
                    -0.145104175534114,
                    0.158296008563706,
                    -0.17726913686089,
                    -0.490234208863142,
                ],
                [
                    -2.32331042514147,
                    -0.693885541424587,
                    -0.0522809641900516,
                    2.00275792547641,
                    -2.4382946997184,
                    0.844783793560244,
                ],
                [
                    -2.2113029795395,
                    -0.219978939390016,
                    -0.268111463764865,
                    0.822218976511082,
                    -1.98165218974949,
                    0.898793867286525,
                ],
                [
                    -2.64534054605335,
                    0.353499075725611,
                    -0.231886560450773,
                    0.538031992369012,
                    -1.9208792787571,
                    -1.17732615796484,
                ],
            ],
        },
        "t": {
            "scale": 1.3,
            "matrix": [
                [
                    -1.24515153038474,
                    0.141342756183746,
                    -0.700054522676855,
                    -0.971731448763251,
                    -0.0562707321438233,
                    -0.281353660719117,
                ],
                [
                    -3.40650347536028,
                    1.03151862464183,
                    0.0108373494470486,
                    2.0945558739255,
                    -2.99701972890206,
                    -0.999006576300688,
                ],
                [
                    -2.48201680538681,
                    0.991206014828623,
                    -0.0113232839848003,
                    2.08153263114011,
                    -2.13146831351308,
                    -1.0149849111967,
                ],
                [
                    -1.65956819982735,
                    -0.922729395500274,
                    -0.160246469529834,
                    1.15341174437534,
                    -1.40379685301571,
                    1.12303748241257,
                ],
                [
                    -1.69992119226441,
                    0.186616714074185,
                    -0.667772732846104,
                    0.926000072617806,
                    -1.99107504312534,
                    -0.805230270976876,
                ],
                [
                    -3.31241192322723,
                    1.01985696134845,
                    0.0244256206111155,
                    -2.2738655272167,
                    1.75128328787636,
                    4.42586960579333,
                ],
            ],
        },
    },
    "tails": {
        "gaussian": {
            "time": [109, 91, 109, 90.5],
            "time2": [109, 91, 109.5, 91],
            "status": [0, 2, 3, 3],
            "eta": [100, 100, 100, 100],
            "matrix": [
                [
                    -43.6281491133321,
                    9.10852310500287,
                    -0.988485209345271,
                    81.9767079450258,
                    -162.044009901993,
                    -18.0048899891103,
                ],
                [
                    -43.6281491133321,
                    -9.10852310500287,
                    -0.988485209345271,
                    81.9767079450258,
                    -162.044009901993,
                    18.0048899891103,
                ],
                [
                    -43.6374914145724,
                    9.1038814367623,
                    -0.990791766363799,
                    -81.8898654482615,
                    1.09829355625334,
                    1472.99239242629,
                ],
                [
                    -43.6374914145724,
                    -9.1038814367623,
                    -0.990791766363799,
                    -81.8898654482615,
                    1.09829355625334,
                    -1472.99239242629,
                ],
            ],
        },
        "lognormal": {
            "time": [8103.08392757538, 0.00012340980408668, 8103.08392757538, 7.48518298877006e-05],
            "time2": [8103.08392757538, 0.00012340980408668, 13359.7268296619, 0.00012340980408668],
            "status": [0, 2, 3, 3],
            "eta": [0, 0, 0, 0],
            "matrix": [
                [
                    -43.6281491133321,
                    9.10852310500287,
                    -0.988485209345271,
                    81.9767079450258,
                    -162.044009901993,
                    -18.0048899891103,
                ],
                [
                    -43.6281491133321,
                    -9.10852310500287,
                    -0.988485209345271,
                    81.9767079450258,
                    -162.044009901993,
                    18.0048899891103,
                ],
                [
                    -43.6374914145724,
                    9.1038814367623,
                    -0.990791766363799,
                    -81.8898654482615,
                    1.09829355625334,
                    1472.99239242629,
                ],
                [
                    -43.6374914145724,
                    -9.1038814367623,
                    -0.990791766363799,
                    -81.8898654482615,
                    1.09829355625334,
                    -1472.99239242629,
                ],
            ],
        },
        "logistic": {
            "time": [140, 60, 140, 59.5],
            "time2": [140, 60, 140.5, 60],
            "status": [0, 2, 3, 3],
            "eta": [100, 100, 100, 100],
            "matrix": [
                [-40, 1, 0, 40, -40, -1],
                [-40, -1, 0, 40, -40, 1],
                [-40.9327521295672, 1, 0, -39.2292529587316, 38.2498284364735, 77.4585059174632],
                [-40.9327521295672, -1, 0, -39.2292529587316, 38.2498284364733, -77.4585059174632],
            ],
        },
        "loglogistic": {
            "time": [
                2.3538526683702e17,
                4.24835425529159e-18,
                2.3538526683702e17,
                2.57675710915498e-18,
            ],
            "time2": [
                2.3538526683702e17,
                4.24835425529159e-18,
                3.8808469624362e17,
                4.24835425529159e-18,
            ],
            "status": [0, 2, 3, 3],
            "eta": [0, 0, 0, 0],
            "matrix": [
                [-40, 1, 0, 40, -40, -1],
                [-40, -1, 0, 40, -40, 1],
                [-40.9327521295672, 1, 0, -39.2292529587316, 38.2498284364735, 77.4585059174632],
                [-40.9327521295672, -1, 0, -39.2292529587316, 38.2498284364733, -77.4585059174632],
            ],
        },
        "extreme": {
            "time": [105, 95, 105, 94.5],
            "time2": [105, 95, 105.5, 95],
            "status": [0, 2, 3, 3],
            "eta": [100, 100, 100, 100],
            "matrix": [
                [
                    -148.413159102577,
                    148.413159102577,
                    -148.413159102583,
                    742.065795512883,
                    -4452.39477307734,
                    -890.478954615493,
                ],
                [
                    -5.00336708183652,
                    -0.996634809825072,
                    -0.00336140685603092,
                    4.98317404912536,
                    -5.06720922052613,
                    1.01344184410523,
                ],
                [
                    -148.413159102577,
                    148.413159102577,
                    -148.413159102583,
                    -742.065795512883,
                    -2968.26318205148,
                    219374.178993452,
                ],
                [
                    -5.93816419592214,
                    -0.994588226508801,
                    -0.00541118776373017,
                    -4.20117194696819,
                    3.07573104886378,
                    -7.33420555090466,
                ],
            ],
        },
        "weibull": {
            "time": [148.413159102577, 0.00673794699908547, 148.413159102577, 0.00408677143846407],
            "time2": [148.413159102577, 0.00673794699908547, 244.69193226422, 0.00673794699908547],
            "status": [0, 2, 3, 3],
            "eta": [0, 0, 0, 0],
            "matrix": [
                [
                    -148.413159102577,
                    148.413159102577,
                    -148.413159102583,
                    742.065795512883,
                    -4452.39477307734,
                    -890.478954615493,
                ],
                [
                    -5.00336708183652,
                    -0.996634809825072,
                    -0.00336140685603092,
                    4.98317404912536,
                    -5.06720922052613,
                    1.01344184410523,
                ],
                [
                    -148.413159102577,
                    148.413159102577,
                    -148.413159102583,
                    -742.065795512883,
                    -2968.26318205148,
                    219374.178993452,
                ],
                [
                    -5.93816419592214,
                    -0.994588226508801,
                    -0.00541118776373017,
                    -4.20117194696819,
                    3.07573104886378,
                    -7.33420555090466,
                ],
            ],
        },
        "t": {
            "time": [140, 60, 140, 59.5],
            "time2": [140, 60, 140.5, 60],
            "status": [0, 2, 3, 3],
            "eta": [100, 100, 100, 100],
            "matrix": [
                [
                    -16.2008273531014,
                    0.124666188557497,
                    0.00310001448587325,
                    4.98664754229988,
                    -0.0266243649026769,
                    -0.000665609122566968,
                ],
                [
                    -16.2008273531014,
                    -0.124666188557497,
                    0.00310001448587325,
                    4.98664754229988,
                    -0.0266243649026769,
                    0.000665609122566968,
                ],
                [
                    -19.0130894242955,
                    0.148622911413131,
                    0.00367042387168503,
                    -4.98153563304214,
                    9.92625619968346,
                    1.47982582231191,
                ],
                [
                    -19.0130894242955,
                    -0.148622911413131,
                    0.00367042387168503,
                    -4.98153563304214,
                    9.92625619968346,
                    -1.47982582231191,
                ],
            ],
        },
    },
    "fit_data": {
        "time": [1.1, 2.3, 3.2, 1.8, 4.5, 2.7, 5.2, 3.9, 6.3, 4.1, 7.2, 5.6],
        "status": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        "x": [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2],
    },
    "fits": {
        "gaussian": {
            "coefficients": [4.11454247116926, 2.14197823723548],
            "scale": 1.26136132936662,
            "working": [
                -0.872564233933778,
                1.52397625603177,
                0.370644471172029,
                -1.45775117627507,
                2.08920335212836,
                -1.41454247116926,
                0.657061881383643,
                1.05457350559895,
                0.90027058648945,
                -1.72812506095765,
                2.17760085475027,
                -1.08491635585184,
            ],
        },
        "weibull": {
            "coefficients": [1.45453837305563, 0.565843989936606],
            "scale": 0.303955846259666,
            "working": [
                -3.83036363261546,
                0.303955846259666,
                0.0445032979333186,
                -2.19546377753408,
                0.303955846259666,
                -1.08248195205263,
                0.0710682105799901,
                0.303955846259666,
                0.0431219760087157,
                -1.25137548887513,
                0.303955846259666,
                -0.87024580351793,
            ],
        },
        "logistic": {
            "coefficients": [4.11109720300906, 2.14987900288137],
            "scale": 0.795726595016932,
            "working": [
                -1.03948089692005,
                1.50529078315271,
                0.393303751543347,
                -2.40036561215541,
                3.02259389168245,
                -2.27607872155741,
                0.736858093485134,
                1.00283754475384,
                1.10279061435804,
                -3.4581814731554,
                3.38553483687835,
                -1.46632207254684,
            ],
        },
        "lognormal": {
            "coefficients": [1.29593175564164, 0.646849039629507],
            "scale": 0.387421981401673,
            "working": [
                -0.553772536207806,
                0.517315934773352,
                0.255328477941747,
                -0.449405474887716,
                0.702129502509204,
                -0.302679982631355,
                0.223357062019842,
                0.383912963392285,
                0.156508453978145,
                -0.402424013634982,
                0.503650319179002,
                -0.349384005455943,
            ],
        },
    },
}


def _assert_reference_matrix(actual, expected):
    assert len(actual) == len(expected)
    for row, reference in zip(actual, expected, strict=True):
        assert len(row) == 6
        assert all(math.isfinite(value) for value in row)
        assert row == pytest.approx(reference, rel=2e-10, abs=2e-10)


@pytest.mark.parametrize("distribution", _REFERENCE["matrices"])
def test_survreg_residual_matrix_matches_r_for_all_censoring_types(distribution):
    data = _REFERENCE["moderate"]
    reference = _REFERENCE["matrices"][distribution]
    matrix = survival.survreg_residual_matrix(
        data["time"],
        data["status"],
        data["eta"],
        reference["scale"],
        distribution,
        time2=data["time2"],
        distribution_parameter=5.0 if distribution == "t" else None,
    )

    _assert_reference_matrix(matrix, reference["matrix"])


@pytest.mark.parametrize(
    ("alias", "reference_distribution"),
    [
        ("extreme_value", "extreme"),
        ("normal", "gaussian"),
        ("loggaussian", "lognormal"),
        ("log-logistic", "loglogistic"),
        ("student_t", "t"),
    ],
)
def test_survreg_residual_matrix_aliases_match_r(alias, reference_distribution):
    data = _REFERENCE["moderate"]
    reference = _REFERENCE["matrices"][reference_distribution]
    matrix = survival.survreg_residual_matrix(
        data["time"],
        data["status"],
        data["eta"],
        reference["scale"],
        alias,
        time2=data["time2"],
        distribution_parameter=5.0 if reference_distribution == "t" else None,
    )

    _assert_reference_matrix(matrix, reference["matrix"])


@pytest.mark.parametrize("distribution", _REFERENCE["tails"])
def test_survreg_residual_matrix_preserves_censored_tail_probabilities(distribution):
    # Gaussian z=+/-9 and logistic z=+/-40 distinguish survival-tail evaluation
    # from subtracting a rounded CDF from one. Both interval orientations are
    # included, as are the corresponding distributions on log-transformed time.
    reference = _REFERENCE["tails"][distribution]
    matrix = survival.survreg_residual_matrix(
        reference["time"],
        reference["status"],
        reference["eta"],
        1.0,
        distribution,
        time2=reference["time2"],
        distribution_parameter=5.0 if distribution == "t" else None,
    )

    _assert_reference_matrix(matrix, reference["matrix"])


@pytest.mark.parametrize("distribution", _REFERENCE["fits"])
def test_survreg_formula_working_residuals_match_r(distribution):
    reference = _REFERENCE["fits"][distribution]
    fit = survival.survreg(
        "Surv(time, status) ~ x",
        data=_REFERENCE["fit_data"],
        dist=distribution,
        max_iter=100,
        eps=1e-12,
    )

    # Confirm the residual comparison starts from the same fitted model. The
    # looser tolerance here accounts for separate optimizer stopping behavior.
    assert fit.location_coefficients == pytest.approx(reference["coefficients"], rel=2e-6, abs=3e-7)
    assert fit.scale == pytest.approx(reference["scale"], rel=2e-6, abs=3e-7)
    assert survival.r_api.residuals(fit, type="working") == pytest.approx(
        reference["working"], rel=2e-6, abs=3e-7
    )
    assert fit.residuals("working").residuals == pytest.approx(
        reference["working"], rel=2e-6, abs=3e-7
    )


def test_extreme_narrow_interval_resolves_rapid_density_change():
    lower = 15.0
    upper = lower + 1e-5
    row = survival.survreg_residual_matrix([lower], [3], [0.0], 1.0, "extreme", time2=[upper])[0]

    # The extreme-value survival function is exp(-exp(z)). Its interval
    # probability has this closed form even when each endpoint underflows.
    expected = -math.exp(lower) + math.log(
        -math.expm1(-math.exp(lower) * math.expm1(upper - lower))
    )
    assert row[0] == pytest.approx(expected, rel=1e-14, abs=1e-8)


@pytest.mark.parametrize(
    ("time", "status", "eta", "sign"),
    [(100000.0, 0, 0.0, 1.0), (1.0, 2, 100001.0, -1.0)],
    ids=["right", "left"],
)
def test_gaussian_far_tail_retains_curvature_and_working_direction(time, status, eta, sign):
    row = survival.survreg_residual_matrix([time], [status], [eta], 1.0, "gaussian")[0]
    working = survival.residuals_survreg(
        [time], [status], [eta], 1.0, "gaussian", residual_type="working"
    ).residuals[0]

    # At |z|=1e5 the normal Mills expansion gives h=z+1/z+O(z^-3),
    # h'=1-1/z^2+O(z^-4), and h/h'=z+2/z+O(z^-3).
    # The omitted terms are below 1e-14, much smaller than these tolerances.
    assert row[1] == pytest.approx(sign * 100000.00001, rel=2e-12)
    assert row[2] == pytest.approx(-0.9999999999, rel=2e-12)
    assert math.isfinite(working)
    assert working == pytest.approx(sign * 100000.00002, rel=2e-12)


def test_lognormal_adjacent_interval_preserves_transformed_width():
    lower = 1e6
    upper = math.nextafter(lower, math.inf)
    row = survival.survreg_residual_matrix([lower], [3], [0.0], 1.0, "lognormal", time2=[upper])[0]

    # The independently transformed endpoints round to the same float. The
    # original interval still has a positive, representable probability.
    z = math.log(lower)
    width = math.log1p((upper - lower) / lower)
    expected = -0.5 * z * z - 0.5 * math.log(math.tau) + math.log(width)
    assert math.isfinite(row[0])
    # The density changes by less than 2e-15 across this interval.
    assert row[0] == pytest.approx(expected, rel=1e-14, abs=2e-12)


def test_gaussian_far_tail_interval_preserves_location_derivatives():
    lower = 100000.0
    censored = survival.survreg_residual_matrix([lower], [0], [0.0], 1.0, "gaussian")[0]
    interval = survival.survreg_residual_matrix(
        [lower], [3], [0.0], 1.0, "gaussian", time2=[lower + 1.0]
    )[0]

    # The excluded upper tail is less than exp(-100000) of the lower tail.
    # The interval's log likelihood and location derivatives therefore equal
    # the right-censored values to far better than double precision.
    assert interval[:3] == pytest.approx(censored[:3], rel=2e-12, abs=2e-12)


def test_student_t_flat_tail_interval_preserves_probability():
    row = survival.survreg_residual_matrix(
        [1e10], [3], [0.0], 1.0, "t", time2=[1e10 + 1.0], distribution_parameter=4.0
    )[0]

    # For df=4, f(z)=(3/8)*(1+z^2/4)^(-5/2). The midpoint-density integral
    # differs from the exact interval probability by less than 2e-20 here.
    # This recorded log probability avoids subtracting two nearly equal CDFs.
    assert row[0] == pytest.approx(-112.64434800016427, rel=0.0, abs=1e-8)
