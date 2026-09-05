"""Public fitted pseudo-values checked against R survival 3.8.11.

``R_FIXTURE_GENERATOR`` reproduces ``R_EXPECTED`` with R and jsonlite. The
tests use the recorded numbers and do not require an R installation.
"""

from contextlib import nullcontext

import pytest

from .helpers import setup_survival_import

survival = setup_survival_import()

R_FIXTURE_GENERATOR = r"""
library(survival)
stopifnot(as.character(packageVersion("survival")) == "3.8.11")
d <- data.frame(time=1:5, status=c(1,0,1,0,1),
                w=c(1,2,1,3,2), id=c("b","a","b","c","a"))
types <- c("survival", "cumhaz", "rmst")
collect <- function(fit, collapse=TRUE, types=c("survival","cumhaz","rmst"), times=c(2,4)) {
    ans <- lapply(types, function(type) {
        unname(suppressWarnings(pseudo(fit, times=times, type=type,
                                      collapse=collapse)))
    })
    names(ans) <- types
    ans
}
result <- list()
result$unweighted <- collect(survfit(Surv(time,status)~1, d), types=types)
result$weighted <- collect(survfit(Surv(time,status)~1, d, weights=w), types=types)
result$weighted_uncollapsed <- collect(survfit(Surv(time,status)~1, d, weights=w),
                                       collapse=FALSE, types=types)
result$ids <- collect(survfit(Surv(time,status)~1, d, id=id), types=types)
result$ids_uncollapsed <- collect(survfit(Surv(time,status)~1, d, id=id),
                                 collapse=FALSE, types=types)
result$weighted_ids <- collect(survfit(Surv(time,status)~1, d, weights=w, id=id),
                              types=types)
result$weighted_ids_uncollapsed <- collect(
    survfit(Surv(time,status)~1, d, weights=w, id=id), collapse=FALSE, types=types)
result$fh <- collect(survfit(Surv(time,status)~1, d, weights=w, stype=2, ctype=1),
                    types=types)
tied <- data.frame(time=c(1,2,2,3,4,5), status=c(1,1,1,0,1,1), w=c(1,2,1,3,2,1))
result$tied_km <- collect(survfit(Surv(time,status)~1, tied, weights=w, stype=1, ctype=2),
                         types=types)
result$tied_fh <- collect(survfit(Surv(time,status)~1, tied, weights=w, stype=2, ctype=2),
                         types=types)
result$start <- collect(survfit(Surv(time,status)~1, d, start.time=2), types=types)
grouped <- data.frame(time=rep(1:5,each=2), status=c(1,0,0,1,1,1,0,0,1,1),
                      w=c(1,2,2,1,1,3,3,1,2,2), group=rep(c("A","B"),5),
                      id=c("b","e","a","d","b","e","c","f","a","d"))
gfit <- survfit(Surv(time,status)~group, grouped, weights=w, id=id)
for (collapse in c(TRUE,FALSE)) {
    name <- if (collapse) "grouped" else "grouped_uncollapsed"
    result[[name]] <- setNames(lapply(types, function(type) {
        frame <- pseudo(gfit, times=c(2,4), type=type, collapse=collapse, data.frame=TRUE)
        setNames(lapply(1:2, function(curve) {
            unname(matrix(frame$pseudo[frame$curve == curve], ncol=2))
        }), c("A","B"))
    }), types)
}
censored <- data.frame(time=1:5,status=rep(0,5))
result$censored <- collect(survfit(Surv(time,status)~1,censored),
                          types=c("survival","cumhaz"))
near <- data.frame(time=c(1,1+1e-10,2,3,4),status=c(1,0,1,0,1),w=c(1,2,1,3,2))
result$near <- collect(survfit(Surv(time,status)~1,near,weights=w), types=types)
residual_fit <- survfit(Surv(time,status)~1,d,weights=w,id=id)
for (mode in c("unweighted", "weighted", "collapsed")) {
    result[[paste0("residual_",mode)]] <- setNames(lapply(types, function(type) {
        unname(residuals(residual_fit, times=c(2,4), type=type,
                         weighted=(mode != "unweighted"), collapse=(mode == "collapsed")))
    }), types)
}
result$cluster <- collect(survfit(Surv(time,status)~1,d,weights=w,id=id,
                                cluster=seq_len(nrow(d))))
near_queries <- c(2-1e-9,2+1e-9)
query_data <- data.frame(time=1:5,status=c(0,1,1,0,1))
query_fit <- survfit(Surv(time,status)~1,query_data)
result$query_pseudo <- collect(query_fit,times=near_queries)
result$query_residual <- setNames(lapply(types,function(type) {
    unname(residuals(query_fit,times=near_queries,type=type))
}),types)
result$time0 <- collect(survfit0(survfit(Surv(time,status)~1,d,start.time=2.5)),
                       times=c(2.5,4))
cat(jsonlite::toJSON(result, digits=16, pretty=TRUE, auto_unbox=TRUE), "\n")
"""

R_EXPECTED = {
    "unweighted": {
        "survival": [
            [-1.1102230246251565e-16, 0],
            [1, 0.6666666666666666],
            [1, -0.2222222222222222],
            [1, 1.1111111111111112],
            [1, 1.1111111111111112],
        ],
        "cumhaz": [
            [1, 1.3333333333333335],
            [0, 0.3333333333333333],
            [0, 1.4444444444444442],
            [0, -0.2222222222222222],
            [0, -0.2222222222222222],
        ],
        "rmst": [
            [1, 0.9999999999999996],
            [2, 3.666666666666667],
            [2, 2.7777777777777772],
            [2, 4.111111111111112],
            [2, 4.111111111111112],
        ],
    },
    "weighted": {
        "survival": [
            [0.3950617283950617, 0.32921810699588483],
            [1.0123456790123457, 0.8436213991769547],
            [0.9506172839506173, 0.1748971193415637],
            [1.074074074074074, 1.265432098765432],
            [1.0123456790123457, 1.0905349794238683],
        ],
        "cumhaz": [
            [0.6049382716049383, 0.7716049382716049],
            [-0.012345679012345678, 0.154320987654321],
            [0.04938271604938271, 0.9104938271604939],
            [-0.07407407407407407, -0.32407407407407407],
            [-0.012345679012345678, -0.12345679012345678],
        ],
        "rmst": [
            [1.3950617283950617, 2.1193415637860076],
            [2.0123456790123457, 3.868312757201646],
            [1.9506172839506173, 3.076131687242798],
            [2.074074074074074, 4.41358024691358],
            [2.0123456790123457, 4.11522633744856],
        ],
    },
    "weighted_uncollapsed": {
        "survival": [
            [0.3950617283950617, 0.32921810699588483],
            [0.9506172839506173, 0.7921810699588476],
            [0.9506172839506173, 0.1748971193415637],
            [0.9506172839506173, 0.9156378600823045],
            [0.9506172839506173, 0.9156378600823045],
        ],
        "cumhaz": [
            [0.6049382716049383, 0.7716049382716049],
            [0.04938271604938271, 0.2160493827160494],
            [0.04938271604938271, 0.9104938271604939],
            [0.04938271604938271, 0.0771604938271605],
            [0.04938271604938271, 0.0771604938271605],
        ],
        "rmst": [
            [1.3950617283950617, 2.1193415637860076],
            [1.9506172839506173, 3.693415637860082],
            [1.9506172839506173, 3.076131687242798],
            [1.9506172839506173, 3.816872427983539],
            [1.9506172839506173, 3.816872427983539],
        ],
    },
    "ids": {
        "survival": [[0.43999999999999995, -0.24], [1.04, 0.96], [0.92, 0.88]],
        "cumhaz": [
            [0.56, 1.56],
            [-0.03999999999999998, -0.040000000000000036],
            [0.08000000000000002, 0.07999999999999996],
        ],
        "rmst": [[1.44, 1.6399999999999992], [2.04, 4.04], [1.9200000000000002, 3.72]],
    },
    "ids_uncollapsed": {
        "survival": [
            [0.31999999999999995, 0.21333333333333332],
            [0.92, 0.6133333333333333],
            [0.92, 0.07999999999999996],
            [0.92, 0.88],
            [0.92, 0.88],
        ],
        "cumhaz": [
            [0.6799999999999999, 1.0133333333333332],
            [0.08000000000000002, 0.41333333333333333],
            [0.08000000000000002, 1.08],
            [0.08000000000000002, 0.07999999999999996],
            [0.08000000000000002, 0.07999999999999996],
        ],
        "rmst": [
            [1.32, 1.853333333333333],
            [1.9200000000000002, 3.453333333333333],
            [1.9200000000000002, 2.9199999999999995],
            [1.9200000000000002, 3.72],
            [1.9200000000000002, 3.72],
        ],
    },
    "weighted_ids": {
        "survival": [
            [0.6296296296296295, 0.154320987654321],
            [1.037037037037037, 1.0123456790123457],
            [1, 1.0555555555555554],
        ],
        "cumhaz": [
            [0.37037037037037035, 0.9537037037037037],
            [-0.037037037037037035, -0.03703703703703698],
            [0, -0.08333333333333331],
        ],
        "rmst": [
            [1.6296296296296295, 2.41358024691358],
            [2.037037037037037, 4.086419753086419],
            [2, 4.055555555555555],
        ],
    },
    "weighted_ids_uncollapsed": {
        "survival": [
            [0.5925925925925926, 0.49382716049382713],
            [0.9259259259259258, 0.7716049382716049],
            [0.9259259259259258, 0.4012345679012345],
            [0.9259259259259258, 0.845679012345679],
            [0.9259259259259258, 0.845679012345679],
        ],
        "cumhaz": [
            [0.4074074074074074, 0.5740740740740741],
            [0.07407407407407407, 0.24074074074074076],
            [0.07407407407407407, 0.6574074074074074],
            [0.07407407407407407, 0.15740740740740744],
            [0.07407407407407407, 0.15740740740740744],
        ],
        "rmst": [
            [1.5925925925925926, 2.679012345679012],
            [1.9259259259259258, 3.6234567901234565],
            [1.9259259259259258, 3.2530864197530858],
            [1.9259259259259258, 3.6975308641975304],
            [1.9259259259259258, 3.6975308641975304],
        ],
    },
    "fh": {
        "survival": [
            [0.4529433578936934, 0.3834082748676003],
            [1.005313306544539, 0.850979341779308],
            [0.9500763116794544, 0.278204784812466],
            [1.0605503014096234, 1.2133469186358814],
            [1.005313306544539, 1.0613863218895765],
        ],
        "cumhaz": [
            [0.6049382716049383, 0.7716049382716049],
            [-0.012345679012345678, 0.154320987654321],
            [0.04938271604938271, 0.9104938271604939],
            [-0.07407407407407407, -0.32407407407407407],
            [-0.012345679012345678, -0.12345679012345678],
        ],
        "rmst": [
            [1.4529433578936934, 2.2892949906549873],
            [2.0053133065445388, 3.8616059548683856],
            [1.9500763116794544, 3.1783574081713746],
            [2.0605503014096236, 4.334447521455128],
            [2.0053133065445388, 4.072012934978654],
        ],
    },
    "tied_km": {
        "survival": [
            [0.24000000000000005, 0.08],
            [-0.12, -0.04000000000000001],
            [0.24, 0.07999999999999999],
            [1.4147368421052633, 0.4715789473684211],
            [1.1431578947368422, -0.4189473684210526],
            [0.871578947368421, 1.0905263157894736],
        ],
        "cumhaz": [
            [1.0066666666666668, 1.6733333333333333],
            [1.191111111111111, 1.8577777777777778],
            [0.8288888888888888, 1.4955555555555555],
            [-0.44666666666666666, 0.21999999999999997],
            [-0.14222222222222225, 1.8577777777777775],
            [0.1622222222222222, -0.5044444444444445],
        ],
        "rmst": [
            [1.3599999999999999, 1.8400000000000003],
            [2.02, 1.7800000000000002],
            [1.96, 2.4400000000000004],
            [2.08, 4.909473684210527],
            [2.02, 4.306315789473684],
            [1.96, 3.703157894736842],
        ],
    },
    "tied_fh": {
        "survival": [
            [0.28846097922560576, 0.1481008049073309],
            [0.1727978812752422, 0.08871739037927073],
            [0.39994348327414914, 0.20533783095847333],
            [1.1998304498224472, 0.61601349287542],
            [1.0089166616393168, 0.08871739037927079],
            [0.8180028734561865, 0.8492543740338252],
        ],
        "cumhaz": [
            [1.0066666666666668, 1.6733333333333333],
            [1.191111111111111, 1.8577777777777778],
            [0.8288888888888888, 1.4955555555555555],
            [-0.44666666666666666, 0.21999999999999997],
            [-0.14222222222222225, 1.8577777777777775],
            [0.1622222222222222, -0.5044444444444445],
        ],
        "rmst": [
            [1.4162252122965413, 1.993147170747753],
            [2.0134179082002746, 2.359013670750759],
            [1.9591276631181171, 2.7590146296664155],
            [2.067708153282432, 4.467369052927327],
            [2.0134179082002746, 4.031251231478908],
            [1.9591276631181171, 3.59513341003049],
        ],
    },
    "start": {
        "survival": [
            [1, 0.6666666666666666],
            [1, 0.6666666666666666],
            [1, -0.22222222222222232],
            [1, 1.1111111111111112],
            [1, 1.1111111111111112],
        ],
        "cumhaz": [
            [0, 0.3333333333333333],
            [0, 0.3333333333333333],
            [0, 1.222222222222222],
            [0, -0.1111111111111111],
            [0, -0.1111111111111111],
        ],
        "rmst": [
            [0, 1.6666666666666665],
            [0, 1.6666666666666665],
            [0, 0.7777777777777778],
            [0, 2.1111111111111107],
            [0, 2.1111111111111107],
        ],
    },
    "grouped": {
        "survival": {
            "A": [
                [0.6296296296296295, 0.154320987654321],
                [1.037037037037037, 1.0123456790123457],
                [1, 1.0555555555555554],
            ],
            "B": [
                [1.0408163265306123, -0.1224489795918367],
                [0.6122448979591837, 0.7346938775510203],
                [0.9183673469387754, 0.673469387755102],
            ],
        },
        "cumhaz": {
            "A": [
                [0.37037037037037035, 0.9537037037037037],
                [-0.037037037037037035, -0.03703703703703698],
                [0, -0.08333333333333331],
            ],
            "B": [
                [-0.040816326530612235, 1.2091836734693877],
                [0.3877551020408163, 0.3877551020408163],
                [0.08163265306122448, 0.33163265306122447],
            ],
        },
        "rmst": {
            "A": [
                [1.6296296296296295, 2.41358024691358],
                [2.037037037037037, 4.086419753086419],
                [2, 4.055555555555555],
            ],
            "B": [[2, 2.9183673469387754], [2, 3.346938775510204], [2, 3.591836734693877]],
        },
    },
    "grouped_uncollapsed": {
        "survival": {
            "A": [
                [0.5925925925925926, 0.49382716049382713],
                [0.9259259259259258, 0.7716049382716049],
                [0.9259259259259258, 0.4012345679012345],
                [0.9259259259259258, 0.845679012345679],
                [0.9259259259259258, 0.845679012345679],
            ],
            "B": [
                [0.8571428571428571, 0.42857142857142855],
                [0.4897959183673469, 0.24489795918367346],
                [0.9183673469387754, 0.24489795918367346],
                [0.9183673469387754, 0.673469387755102],
                [0.9183673469387754, 0.673469387755102],
            ],
        },
        "cumhaz": {
            "A": [
                [0.4074074074074074, 0.5740740740740741],
                [0.07407407407407407, 0.24074074074074076],
                [0.07407407407407407, 0.6574074074074074],
                [0.07407407407407407, 0.15740740740740744],
                [0.07407407407407407, 0.15740740740740744],
            ],
            "B": [
                [0.14285714285714285, 0.6428571428571428],
                [0.510204081632653, 1.010204081632653],
                [0.08163265306122448, 0.8316326530612244],
                [0.08163265306122448, 0.33163265306122447],
                [0.08163265306122448, 0.33163265306122447],
            ],
        },
        "rmst": {
            "A": [
                [1.5925925925925926, 2.679012345679012],
                [1.9259259259259258, 3.6234567901234565],
                [1.9259259259259258, 3.2530864197530858],
                [1.9259259259259258, 3.6975308641975304],
                [1.9259259259259258, 3.6975308641975304],
            ],
            "B": [
                [2, 3.2857142857142856],
                [2, 2.7346938775510203],
                [2, 3.1632653061224487],
                [2, 3.591836734693877],
                [2, 3.591836734693877],
            ],
        },
    },
    "censored": {
        "survival": [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],
        "cumhaz": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    },
    "near": {
        "survival": [
            [0.32921810699588483, 0],
            [0.8436213991769547, 0],
            [0.1748971193415637, 0],
            [1.265432098765432, 0],
            [1.0905349794238683, 0],
        ],
        "cumhaz": [
            [0.7716049382716049, 1.7716049382716048],
            [0.154320987654321, 1.1543209876543208],
            [0.9104938271604939, 1.9104938271604937],
            [-0.32407407407407407, 0.6759259259259258],
            [-0.12345679012345678, 0.8765432098765435],
        ],
        "rmst": [
            [1.3950617283950617, 2.0534979423868314],
            [2.0123456790123457, 3.699588477366255],
            [1.9506172839506173, 2.3004115226337447],
            [2.074074074074074, 4.604938271604938],
            [2.0123456790123457, 4.193415637860082],
        ],
    },
    "residual_unweighted": {
        "survival": [
            [-0.09876543209876543, -0.08230452674897118],
            [0.012345679012345678, 0.010288065843621397],
            [0.012345679012345678, -0.11316872427983539],
            [0.012345679012345678, 0.03497942386831276],
            [0.012345679012345678, 0.03497942386831276],
        ],
        "cumhaz": [
            [0.09876543209876543, 0.09876543209876543],
            [-0.012345679012345678, -0.012345679012345678],
            [-0.012345679012345678, 0.12654320987654322],
            [-0.012345679012345678, -0.040123456790123455],
            [-0.012345679012345678, -0.040123456790123455],
        ],
        "rmst": [
            [-0.09876543209876543, -0.27983539094650206],
            [0.012345679012345678, 0.03497942386831276],
            [0.012345679012345678, -0.08847736625514407],
            [0.012345679012345678, 0.05967078189300413],
            [0.012345679012345678, 0.05967078189300413],
        ],
    },
    "residual_weighted": {
        "survival": [
            [-0.09876543209876543, -0.08230452674897118],
            [0.024691358024691357, 0.020576131687242795],
            [0.012345679012345678, -0.11316872427983539],
            [0.037037037037037035, 0.10493827160493827],
            [0.024691358024691357, 0.06995884773662552],
        ],
        "cumhaz": [
            [0.09876543209876543, 0.09876543209876543],
            [-0.024691358024691357, -0.024691358024691357],
            [-0.012345679012345678, 0.12654320987654322],
            [-0.037037037037037035, -0.12037037037037036],
            [-0.024691358024691357, -0.08024691358024691],
        ],
        "rmst": [
            [-0.09876543209876543, -0.27983539094650206],
            [0.024691358024691357, 0.06995884773662552],
            [0.012345679012345678, -0.08847736625514407],
            [0.037037037037037035, 0.1790123456790124],
            [0.024691358024691357, 0.11934156378600826],
        ],
    },
    "residual_collapsed": {
        "survival": [
            [-0.08641975308641975, -0.19547325102880658],
            [0.04938271604938271, 0.09053497942386832],
            [0.037037037037037035, 0.10493827160493827],
        ],
        "cumhaz": [
            [0.08641975308641975, 0.22530864197530864],
            [-0.04938271604938271, -0.10493827160493827],
            [-0.037037037037037035, -0.12037037037037036],
        ],
        "rmst": [
            [-0.08641975308641975, -0.36831275720164613],
            [0.04938271604938271, 0.18930041152263377],
            [0.037037037037037035, 0.1790123456790124],
        ],
    },
    "cluster": {
        "survival": [
            [0.5925925925925926, 0.49382716049382713],
            [0.9629629629629629, 0.802469135802469],
            [0.9259259259259258, 0.4012345679012345],
            [1, 1.0555555555555554],
            [0.9629629629629629, 0.9506172839506173],
        ],
        "cumhaz": [
            [0.4074074074074074, 0.5740740740740741],
            [0.037037037037037035, 0.20370370370370372],
            [0.07407407407407407, 0.6574074074074074],
            [0, -0.08333333333333331],
            [0.037037037037037035, 0.03703703703703706],
        ],
        "rmst": [
            [1.5925925925925926, 2.679012345679012],
            [1.9629629629629628, 3.7283950617283947],
            [1.9259259259259258, 3.2530864197530858],
            [2, 4.055555555555555],
            [1.9629629629629628, 3.876543209876543],
        ],
    },
    "query_pseudo": {
        "survival": [[1, 0.75], [1, 0.75], [1, 0.75], [1, 0.75], [1, 0.75]],
        "cumhaz": [[0, 0.25], [0, 0.25], [0, 0.25], [0, 0.25], [0, 0.25]],
        "rmst": [
            [1.999999999, 2.00000000075],
            [1.999999999, 2.00000000075],
            [1.999999999, 2.00000000075],
            [1.999999999, 2.00000000075],
            [1.999999999, 2.00000000075],
        ],
    },
    "query_residual": {
        "survival": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "cumhaz": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        "rmst": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    },
    "time0": {
        "survival": [
            [1, 0.6666666666666666],
            [1, 0.6666666666666666],
            [1, -1.1102230246251565e-16],
            [1, 1],
            [1, 1],
        ],
        "cumhaz": [[0, 0.3333333333333333], [0, 0.3333333333333333], [0, 1], [0, 0], [0, 0]],
        "rmst": [
            [0, 1.1666666666666665],
            [0, 1.1666666666666665],
            [0, 0.5],
            [0, 1.4999999999999998],
            [0, 1.4999999999999998],
        ],
    },
}

_TYPES = ("survival", "cumhaz", "rmst")
_TIMES = [2.0, 4.0]
_WEIGHTS = [1.0, 2.0, 1.0, 3.0, 2.0]
_IDS = ["b", "a", "b", "c", "a"]


def _response():
    return survival.Surv([1.0, 2.0, 3.0, 4.0, 5.0], [1, 0, 1, 0, 1])


def _grouped_data():
    return {
        "time": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        "status": [1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        "weights": [1, 2, 2, 1, 1, 3, 3, 1, 2, 2],
        "group": ["A", "B"] * 5,
        "id": ["b", "e", "a", "d", "b", "e", "c", "f", "a", "d"],
    }


def _assert_matrix(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == pytest.approx(expected_row, rel=2e-12, abs=2e-12)


@pytest.mark.parametrize("pseudo_type", _TYPES)
@pytest.mark.parametrize(
    ("fixture", "options", "collapse"),
    [
        ("unweighted", {}, True),
        ("weighted", {"weights": _WEIGHTS}, True),
        ("weighted_uncollapsed", {"weights": _WEIGHTS}, False),
        ("ids", {"id": _IDS}, True),
        ("ids_uncollapsed", {"id": _IDS}, False),
        ("weighted_ids", {"weights": _WEIGHTS, "id": _IDS}, True),
        ("weighted_ids_uncollapsed", {"weights": _WEIGHTS, "id": _IDS}, False),
        ("fh", {"weights": _WEIGHTS, "stype": 2, "ctype": 1}, True),
        ("start", {"start_time": 2.0}, True),
    ],
)
def test_right_survfit_pseudo_matches_r(fixture, options, collapse, pseudo_type):
    fit = survival.survfit(_response(), model=True, **options)
    actual = survival.pseudo(fit, times=_TIMES, type=pseudo_type, collapse=collapse)
    _assert_matrix(actual, R_EXPECTED[fixture][pseudo_type])


@pytest.mark.parametrize("pseudo_type", _TYPES)
@pytest.mark.parametrize(("fixture", "stype"), [("tied_km", 1), ("tied_fh", 2)])
def test_right_survfit_pseudo_matches_r_tied_hazard_approximation(fixture, stype, pseudo_type):
    response = survival.Surv([1.0, 2.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 1, 0, 1, 1])
    fit = survival.survfit(response, weights=[1, 2, 1, 3, 2, 1], stype=stype, ctype=2, model=True)
    warning = (
        pytest.warns(RuntimeWarning, match="ctype=2.*approximate")
        if pseudo_type == "cumhaz" or (pseudo_type == "survival" and stype == 2)
        else nullcontext()
    )
    with warning:
        actual = survival.pseudo(fit, times=_TIMES, type=pseudo_type)
    _assert_matrix(actual, R_EXPECTED[fixture][pseudo_type])


@pytest.mark.parametrize("pseudo_type", _TYPES)
@pytest.mark.parametrize("collapse", [True, False])
@pytest.mark.parametrize("formula", [True, False])
def test_grouped_weighted_survfit_pseudo_matches_r(pseudo_type, collapse, formula):
    data = _grouped_data()
    if formula:
        fit = survival.survfit(
            "Surv(time, status) ~ group",
            data=data,
            weights=data["weights"],
            id=data["id"],
            model=True,
        )
    else:
        fit = survival.survfit(
            survival.Surv(data["time"], data["status"]),
            group=data["group"],
            weights=data["weights"],
            id=data["id"],
            model=True,
        )
    actual = survival.pseudo(fit, times=_TIMES, type=pseudo_type, collapse=collapse)
    fixture = "grouped" if collapse else "grouped_uncollapsed"
    assert list(actual) == ["A", "B"]
    for label in actual:
        _assert_matrix(actual[label], R_EXPECTED[fixture][pseudo_type][label])


@pytest.mark.parametrize("collapse", [True, False])
def test_right_survfit_pseudo_frame_retains_subject_ids(collapse):
    fit = survival.survfit(_response(), id=_IDS, weights=_WEIGHTS, model=True)
    frame = survival.pseudo(fit, times=[4.0], collapse=collapse, data_frame=True)
    fixture = "weighted_ids" if collapse else "weighted_ids_uncollapsed"
    expected = [row[1] for row in R_EXPECTED[fixture]["survival"]]
    assert frame["id"] == (["b", "a", "c"] if collapse else _IDS)
    assert frame["time"] == [4.0] * len(expected)
    assert frame["pseudo"] == pytest.approx(expected, rel=2e-12, abs=2e-12)


@pytest.mark.parametrize("residual_type", _TYPES)
@pytest.mark.parametrize(
    ("mode", "weighted", "collapse"),
    [("unweighted", False, False), ("weighted", True, False), ("collapsed", True, True)],
)
def test_weighted_right_survfit_residuals_match_r(mode, weighted, collapse, residual_type):
    fit = survival.survfit(_response(), weights=_WEIGHTS, id=_IDS, model=True)
    actual = survival.survfit_residuals(
        fit, times=_TIMES, type=residual_type, weighted=weighted, collapse=collapse
    )
    _assert_matrix(actual["resid"], R_EXPECTED[f"residual_{mode}"][residual_type])
    assert actual["id"] == (["b", "a", "c"] if collapse else _IDS)


@pytest.mark.parametrize("pseudo_type", ["survival", "cumhaz"])
def test_right_survfit_pseudo_without_events_matches_r(pseudo_type):
    fit = survival.survfit(survival.Surv([1, 2, 3, 4, 5], [0] * 5), model=True)
    actual = survival.pseudo(fit, times=_TIMES, type=pseudo_type)
    _assert_matrix(actual, R_EXPECTED["censored"][pseudo_type])


@pytest.mark.parametrize("pseudo_type", _TYPES)
def test_right_survfit_pseudo_uses_fitted_near_ties(pseudo_type):
    response = survival.Surv([1, 1 + 1e-10, 2, 3, 4], [1, 0, 1, 0, 1])
    fit = survival.survfit(response, weights=_WEIGHTS, model=True)
    actual = survival.pseudo(fit, times=_TIMES, type=pseudo_type)
    _assert_matrix(actual, R_EXPECTED["near"][pseudo_type])


def test_direct_vector_pseudo_keeps_its_existing_result_and_rmst_jackknife():
    result = survival.pseudo([1, 2, 3, 4, 5], [1, 0, 1, 0, 1], [2, 4], "rmst")
    assert type(result).__name__ == "PseudoResult"
    assert result.time == _TIMES
    assert result.type_ == "rmst"
    assert result.n == 5
    _assert_matrix(result.pseudo, [[1, 1], [2, 11 / 3], [2, 8 / 3], [2, 25 / 6], [2, 25 / 6]])


@pytest.mark.parametrize("residual_type", _TYPES)
def test_survfit_residuals_collapse_defaults_to_weighted(residual_type):
    fit = survival.survfit(_response(), weights=_WEIGHTS, id=_IDS, model=True)
    result = survival.survfit_residuals(fit, times=_TIMES, type=residual_type, collapse=True)
    _assert_matrix(result["resid"], R_EXPECTED["residual_collapsed"][residual_type])
    assert result["id"] == ["b", "a", "c"]


@pytest.mark.parametrize("residual_type", _TYPES)
@pytest.mark.parametrize("ids", [None, ["a", "b", "c", "d", "e"]])
def test_survfit_residuals_unweighted_allows_ineffective_collapse(residual_type, ids):
    fit = survival.survfit(_response(), weights=_WEIGHTS, id=ids, model=True)
    result = survival.survfit_residuals(
        fit, times=_TIMES, type=residual_type, collapse=True, weighted=False
    )
    _assert_matrix(result["resid"], R_EXPECTED["residual_unweighted"][residual_type])
    assert result["id"] == (list(range(1, 6)) if ids is None else ids)


@pytest.mark.parametrize("pseudo_type", _TYPES)
@pytest.mark.parametrize("collapse", [True, False])
def test_survfit_pseudo_retains_nonstandard_formula_columns(pseudo_type, collapse):
    data = _grouped_data()
    renamed = {
        "futime": data["time"],
        "fustat": data["status"],
        "g": data["group"],
        "subject": data["id"],
    }
    canonical = survival.survfit(
        "Surv(time, status) ~ group",
        data=data,
        weights=data["weights"],
        id=data["id"],
        model=True,
    )
    fit = survival.survfit(
        "Surv(futime, fustat) ~ g",
        data=renamed,
        weights=data["weights"],
        id="subject",
        model=True,
    )
    actual = survival.pseudo(fit, times=_TIMES, type=pseudo_type, collapse=collapse)
    equivalent = survival.pseudo(canonical, times=_TIMES, type=pseudo_type, collapse=collapse)
    fixture = "grouped" if collapse else "grouped_uncollapsed"
    assert list(actual) == list(equivalent) == ["A", "B"]
    for label in actual:
        _assert_matrix(actual[label], R_EXPECTED[fixture][pseudo_type][label])
        _assert_matrix(actual[label], equivalent[label])


@pytest.mark.parametrize("pseudo_type", _TYPES)
@pytest.mark.parametrize("collapse", [True, False])
def test_unique_clusters_keep_repeated_subject_rows_separate(pseudo_type, collapse):
    fit = survival.survfit(
        _response(), weights=_WEIGHTS, id=_IDS, cluster=list(range(5)), model=True
    )
    actual = survival.pseudo(fit, times=_TIMES, type=pseudo_type, collapse=collapse)
    fixture = "cluster" if collapse else "weighted_ids_uncollapsed"
    _assert_matrix(actual, R_EXPECTED[fixture][pseudo_type])
    residuals = survival.survfit_residuals(fit, times=_TIMES, type=pseudo_type, collapse=collapse)
    residual_fixture = "residual_weighted" if collapse else "residual_unweighted"
    _assert_matrix(residuals["resid"], R_EXPECTED[residual_fixture][pseudo_type])
    assert residuals["id"] == _IDS


@pytest.mark.parametrize("pseudo_type", _TYPES)
def test_near_query_times_preserve_pseudo_estimates_and_normalize_residuals(pseudo_type):
    fit = survival.survfit(survival.Surv([1, 2, 3, 4, 5], [0, 1, 1, 0, 1]), model=True)
    times = [2 - 1e-9, 2 + 1e-9]
    pseudo = survival.pseudo(fit, times=times, type=pseudo_type)
    residuals = survival.survfit_residuals(fit, times=times, type=pseudo_type)
    _assert_matrix(pseudo, R_EXPECTED["query_pseudo"][pseudo_type])
    _assert_matrix(residuals["resid"], R_EXPECTED["query_residual"][pseudo_type])


@pytest.mark.parametrize("pseudo_type", _TYPES)
def test_survfit0_preserves_conditional_start_for_pseudo_values(pseudo_type):
    fit = survival.survfit(_response(), start_time=2.5, model=True)
    augmented = survival.survfit0(fit)
    assert augmented.time[0] == augmented.start_time == 2.5
    actual = survival.pseudo(augmented, times=[2.5, 4.0], type=pseudo_type)
    _assert_matrix(actual, R_EXPECTED["time0"][pseudo_type])


@pytest.mark.parametrize("function_name", ["pseudo", "survfit_residuals"])
def test_grouped_survfit_rejects_collapsing_subjects_across_curves(function_name):
    fit = survival.survfit(
        survival.Surv([1, 2, 3, 4, 5, 6], [1, 0, 1, 1, 0, 1]),
        group=["A", "A", "A", "B", "B", "B"],
        id=["a", "b", "a", "a", "c", "a"],
        model=True,
    )
    function = getattr(survival, function_name)
    with pytest.raises(ValueError, match="same id appears in multiple curves, cannot collapse"):
        function(fit, times=[2.0], collapse=True)
    function(fit, times=[2.0], collapse=False)
