#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else "python/tests/fixtures/formula_algebra_reference.json"
d <- expand.grid(a=factor(c("b","a"),levels=c("b","a")),
    b=factor(c("low","mid","high"),levels=c("low","mid","high")),replicate=1:2)
d$x <- sin(seq_len(nrow(d))) + seq_len(nrow(d))/5
d$y <- cos(seq_len(nrow(d)))
d$outcome <- seq_len(nrow(d))
rhs <- c("a*b-b:a","a:b+b:a","a:b+b:a-b:a","a*b*x-x:b:a",
    "(a+b)/x","a/(b/x)","a/(b+x)/y","(a+b)*x/y",
    "a %in% (b+x)","(a+b) %in% (x+y)","(a+b)^2*x","(a+b)^2:x",
    "a+(b-1)","a-(b-1)","a-(b+1)","a-(0+b)","(a-1)+1",
    "0+a*(b+1)","a+1-0","a-1+0","a-(b-1)*x","a+((b-1)*x)",
    "(a+b)^2.5","a^2*b","(a+b)^0","(a+b)^1","a^(2+1)","(a+b)^2^2")
for(left in c("a","(a+b)","(x+a)","(a*b)","(a+b)^2","(a-1)"))
    for(right in c("b","(b+x)","x^2","(b-1)"))
        for(op in c("*","/",":","%in%")) rhs <- c(rhs,paste(left,op,right))
rhs <- unique(rhs)
cases <- lapply(rhs,function(rhs) {
    formula <- paste("outcome ~",rhs)
    tryCatch({
        tt <- terms(as.formula(formula))
        mf <- model.frame(tt,d)
        x <- model.matrix(tt,mf)
        list(formula=formula,intercept=attr(tt,"intercept"),
            terms=I(attr(tt,"term.labels")),variables=I(names(mf)[-1]),
            x=unname(x),columns=I(colnames(x)),assign=I(attr(x,"assign")))
    },error=function(e) list(formula=formula,error=conditionMessage(e)))
})
reference <- list(metadata=list(generator="scripts/generate_formula_algebra_reference.R",
    r_version=as.character(getRversion())),data=d,
    factor_levels=list(a=I(levels(d$a)),b=I(levels(d$b))),cases=cases)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
write_json(reference,output,auto_unbox=TRUE,digits=17,pretty=TRUE,na="null",null="null",dataframe="columns")
cat(length(cases),"formula algebra cases written to",output,"\n")
