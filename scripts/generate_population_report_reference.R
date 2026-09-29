#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
output <- if(length(args)) args[[1]] else 'python/tests/fixtures/population_report_reference.json'
options(digits=7,width=80)
encode <- function(x) if(is.null(x)) NULL else list(rows=I(rownames(x)),columns=I(colnames(x)),values=unname(unclass(x)))
cases <- list(); objects <- list(); specs <- list(); datasets <- list(lung=as.list(lung),cgd=as.list(cgd))
add_object <- function(name,kind,formula,data,arguments=list(),x=NULL) {
  if(is.null(x)) x <- do.call(get(kind),c(list(as.formula(formula),data=as.data.frame(datasets[[data]])),arguments))
  objects[[name]] <<- x
  specs[[name]] <<- list(kind=kind,formula=formula,data=data,arguments=arguments)
}
add_object('right','survcheck','Surv(time,status)~1','lung',list(id=seq_len(nrow(lung))))
add_object('counting','survcheck','Surv(tstart,tstop,status)~1','cgd',list(id=cgd$id))
d <- data.frame(id=c(1,1,1,2,2,3,3,4,4),start=c(0,2,3,0,4,0,2,0,4),stop=c(2,4,6,2,6,2,5,2,6),event=c(1,0,1,1,1,1,0,1,1),istate=c('well','event','event','well','well','well','well','well','well'))
datasets$checks <- as.list(d)
add_object('problems','survcheck','Surv(start,stop,event)~1','checks',list(id=d$id,istate=d$istate))
datasets$missing <- as.list(transform(d,z=c(NA,rep(1,8))))
add_object('missing','survcheck','Surv(start,stop,event)~z','missing',list(id=d$id))
datasets$multistate <- as.list(data.frame(id=c(1,1,2,2,3,3),start=c(0,2,0,3,0,2),stop=c(2,5,3,6,2,4),event=c('eventA','eventB','eventA','censor','eventB','censor')))
add_object('multistate','survcheck','Surv(start,stop,factor(event))~1','multistate',list(id=c(1,1,2,2,3,3)))
datasets$censored <- list(time=1:4,status=rep(0,4),id=1:4)
add_object('no_events','survcheck','Surv(time,status)~1','censored',list(id=1:4))
datasets$timeline <- list(id=c(1,1,1,2,2),time=c(0,2,5,0,4),status=c(0,1,0,0,1))
add_object('timeline','survcheck','Surv2(time,status)~1','timeline',list(id=c(1,1,1,2,2)))
datasets$population <- list(time=c(100,400,900,300),status=c(1,0,1,1),age=c(60,70,65,80)*365.25,sex=c(1,2,1,2),year=as.Date(c('1995-03-01','1996-06-15','1997-01-01','1998-09-09')),race=c('white','black','white','black'),grp=c('a','b','a','b'))
add_object('plain','pyears','time~1','population')
add_object('events','pyears','Surv(time,status)~grp','population')
add_object('cut','pyears','Surv(time,status)~grp+tcut(age,c(0,65,75)*365.25)','population')
add_object('weighted','pyears','Surv(time,status)~grp','population',list(weights=c(.25,1,1.75,2),scale=1))
add_object('frame','pyears','Surv(time,status)~grp','population',list(data.frame=TRUE))
for(name in c('survexp.us','survexp.usr','survexp.mn')) {
  d<-as.data.frame(datasets$population)
  x<-pyears(Surv(time,status)~grp,d,ratetable=get(name))
  add_object(name,'pyears','Surv(time,status)~grp','population',list(ratetable=name),x)
}
add_object('population_weights','pyears','Surv(time,status)~grp','population',list(ratetable='survexp.us',weights=c(.5,2,0,4)),x=pyears(Surv(time,status)~grp,as.data.frame(datasets$population),ratetable=survexp.us,weights=c(.5,2,0,4)))
datasets$population_missing <- datasets$population;datasets$population_missing$age[2] <- NA
add_object('population_missing','pyears','Surv(time,status)~grp','population_missing',list(ratetable='survexp.us',data.frame=TRUE),x=pyears(Surv(time,status)~grp,as.data.frame(datasets$population_missing),ratetable=survexp.us,data.frame=TRUE))
joint <- read_json('python/tests/fixtures/yates_joint_reference.json',simplifyVector=FALSE)
for(name in c('two_factors','pairwise','missing_cell','sgtt','numeric_levels','two_numeric','cox')) {
  spec <- joint$cases[[which(vapply(joint$cases,function(x)x$name==name,logical(1)))]]
  data <- as.data.frame(lapply(spec$data,unlist))
  fit <- if(spec$cox) coxph(as.formula(spec$formula),data) else lm(as.formula(spec$formula),data)
  arguments <- list(fit=fit,term=spec$term,population=spec$population,test=spec$test,method=spec$method)
  if(!is.null(spec$levels)) arguments$levels <- lapply(spec$levels,unlist)
  result <- do.call(yates,arguments)
  add_object(paste0('yates_',name),'yates',NULL,NULL,list(reference=name),result)
}
add <- function(name,object,arguments=list(),width=80) {
  x<-objects[[object]];kind<-specs[[object]]$kind;x$call<-NULL
  input<-if(kind=='survcheck') list(n=I(x$n),transitions=encode(x$transitions),events=encode(x$events),flag=as.list(x$flag),omit=if(is.null(x$na.action)) NULL else I(as.integer(x$na.action)),problems=x[c('overlap','gap','jump','teleport')]) else if(kind=='pyears') list(pyears=x$pyears,event=x$event,offtable=x$offtable,observations=x$observations,data=x$data,summary=x$summary,omit=if(is.null(x$na.action)) NULL else I(as.integer(x$na.action))) else list(estimate=x$estimate,test=encode(x$test))
  saved<-options(width=width)
  lines<-capture.output(do.call(print,c(list(x),arguments)))
  options(saved)
  cases[[length(cases)+1L]]<<-list(name=name,object=object,kind=kind,arguments=arguments,width=width,input=input,lines=I(sub('[[:blank:]]+$','',lines)))
}
for(name in names(objects)) add(name,name)
add('multistate_narrow','multistate',width=25)
add('right_narrow','right',width=25)
add('counting_narrow','counting',width=22)
add('yates_narrow','yates_two_factors',width=35)
add('yates_pairwise_narrow','yates_pairwise',width=25)
add('yates_precision','yates_cox',list(digits=3,dig.tst=2),width=40)
add('yates_threshold','yates_two_numeric',list(eps=.02,dig.tst=5))
add('yates_zero_threshold','yates_numeric_levels',list(eps=0,digits=7))
write_json(list(r_version=R.version.string,survival_version=as.character(packageVersion('survival')),datasets=datasets,objects=specs,cases=cases),output,auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null',null='null',dataframe='columns')
