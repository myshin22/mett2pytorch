opres.exp <- function(t.n, t.a, mu_true, lam, n.interim, rate, FUP, nsim){
  nmax = max(n.interim)
  nobs = n.interim+1
  nobs[length(nobs)] = nmax
  
  out1 <- c()
  pts=ttrial <- c()
  for (sim in 1:nsim){
    wait.t = rexp(nmax,rate = rate)
    arrival.t = cumsum(wait.t)
    event.t = rexp(nmax,rate=log(2)/mu_true)
    tobs = arrival.t[nobs]
    tobs[length(tobs)] = tobs[length(tobs)] + FUP
    
    k=1
    n.fail = sum(arrival.t[1:n.interim[k]] + event.t[1:n.interim[k]] <= tobs[k])
    t.event = rep(0,n.interim[k])
    t.ind = rep(0,n.interim[k])
    for(j in 1:length(t.event)) {
      t.event[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],event.t[j],tobs[k]-arrival.t[j])
      t.ind[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],1,0)
    }
    fit <- survfit(Surv(t.event, t.ind)~1)
    if(min(fit$surv)>0.5){
      phihat <- max(t.event)
    }else{
      phihat <- summary(fit)$table[7] 
    }
    if(phihat <= lam){
      out1[sim] <- 21 
      pts[sim] <- n.interim[k]
      ttrial[sim] <- tobs[k]
    }else{
      k=2
      n.fail = sum(arrival.t[1:n.interim[k]] + event.t[1:n.interim[k]] <= tobs[k])
      t.event = rep(0,n.interim[k])
      t.ind = rep(0,n.interim[k])
      for(j in 1:length(t.event)) {
        t.event[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],event.t[j],tobs[k]-arrival.t[j])
        t.ind[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],1,0)
      }
      fit <- survfit(Surv(t.event, t.ind)~1)
      if(min(fit$surv)>0.5){
        phihat <- max(t.event)
      }else{
        phihat <- summary(fit)$table[7] 
      }
      if(phihat > lam){
        out1[sim] <- 1 
        pts[sim] <- n.interim[k]
        ttrial[sim] <- tobs[k]
      }else{
        out1[sim] <- 2 
        pts[sim] <- n.interim[k]
        ttrial[sim] <- tobs[k]
      }
    }
  }
  
  phat <- length(which(out1==1))/nsim
  earlystop <- length(which(out1==21))/nsim
  mpts <- mean(pts)
  mtrial <- mean(ttrial)
  
  res <-list(phat, earlystop, mpts, mtrial)
  names(res) <- c("phat", "earlystop", "mpts", "mtrial")
  return(res)
}


opres.unif <- function(t.n, t.a, mu_true, lam, n.interim, rate, FUP, 
                       nsim, para1, para2){
  nmax = max(n.interim)
  nobs = n.interim+1
  nobs[length(nobs)] = nmax
  
  out1 <- c()
  pts=ttrial <- c()
  for (sim in 1:nsim){
    wait.t = rexp(nmax,rate = rate)
    arrival.t = cumsum(wait.t)
    event.t = runif(nmax, para1, para2) 
    tobs = arrival.t[nobs]
    tobs[length(tobs)] = tobs[length(tobs)] + FUP
    
    k=1
    n.fail = sum(arrival.t[1:n.interim[k]] + event.t[1:n.interim[k]] <= tobs[k])
    t.event = rep(0,n.interim[k])
    t.ind = rep(0,n.interim[k])
    for(j in 1:length(t.event)) {
      t.event[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],event.t[j],tobs[k]-arrival.t[j])
      t.ind[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],1,0)
    }
    fit <- survfit(Surv(t.event, t.ind)~1)
    if(min(fit$surv)>0.5){
      phihat <- max(t.event)
    }else{
      phihat <- summary(fit)$table[7] 
    }
    if(phihat <= lam){
      out1[sim] <- 21 
      pts[sim] <- n.interim[k]
      ttrial[sim] <- tobs[k]
    }else{
      k=2
      
      n.fail = sum(arrival.t[1:n.interim[k]] + event.t[1:n.interim[k]] <= tobs[k])
      t.event = rep(0,n.interim[k])
      t.ind = rep(0,n.interim[k])
      for(j in 1:length(t.event)) {
        t.event[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],event.t[j],tobs[k]-arrival.t[j])
        t.ind[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],1,0)
      }
      fit <- survfit(Surv(t.event, t.ind)~1)
      if(min(fit$surv)>0.5){
        phihat <- max(t.event)
      }else{
        phihat <- summary(fit)$table[7] 
      }
      if(phihat > lam){
        out1[sim] <- 1 
        pts[sim] <- n.interim[k]
        ttrial[sim] <- tobs[k]
      }else{
        out1[sim] <- 2 
        pts[sim] <- n.interim[k]
        ttrial[sim] <- tobs[k]
      }
    }
  }
  phat <- length(which(out1==1))/nsim
  earlystop <- length(which(out1==21))/nsim
  mpts <- mean(pts)
  mtrial <- mean(ttrial)
  
  res <-list(phat, earlystop, mpts, mtrial)
  names(res) <- c("phat", "earlystop", "mpts", "mtrial")
  return(res)
}


opres.wei <- function(t.n, t.a, mu_true, lam, n.interim, rate, FUP, nsim, para1, para2){
  nmax = max(n.interim)
  nobs = n.interim+1
  nobs[length(nobs)] = nmax
  
  out1 <- c()
  pts=ttrial <- c()
  for (sim in 1:nsim){
    wait.t = rexp(nmax,rate = rate)
    arrival.t = cumsum(wait.t)
    event.t = rweibull(nmax, shape=para1, scale=para2)
    tobs = arrival.t[nobs]
    tobs[length(tobs)] = tobs[length(tobs)] + FUP
    
    k=1
    n.fail = sum(arrival.t[1:n.interim[k]] + event.t[1:n.interim[k]] <= tobs[k])
    t.event = rep(0,n.interim[k])
    t.ind = rep(0,n.interim[k])
    for(j in 1:length(t.event)) {
      t.event[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],event.t[j],tobs[k]-arrival.t[j])
      t.ind[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],1,0)
    }
    fit <- survfit(Surv(t.event, t.ind)~1)
    if(min(fit$surv)>0.5){
      phihat <- max(t.event)
    }else{
      phihat <- summary(fit)$table[7] 
    }
    if(phihat <= lam){
      out1[sim] <- 21 
      pts[sim] <- n.interim[k]
      ttrial[sim] <- tobs[k]
    }else{
      k=2
      
      n.fail = sum(arrival.t[1:n.interim[k]] + event.t[1:n.interim[k]] <= tobs[k])
      t.event = rep(0,n.interim[k])
      t.ind = rep(0,n.interim[k])
      for(j in 1:length(t.event)) {
        t.event[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],event.t[j],tobs[k]-arrival.t[j])
        t.ind[j] = ifelse(arrival.t[j]+event.t[j]<=tobs[k],1,0)
      }
      fit <- survfit(Surv(t.event, t.ind)~1)
      if(min(fit$surv)>0.5){
        phihat <- max(t.event)
      }else{
        phihat <- summary(fit)$table[7] 
      }
      if(phihat > lam){
        out1[sim] <- 1 
        pts[sim] <- n.interim[k]
        ttrial[sim] <- tobs[k]
      }else{
        out1[sim] <- 2 
        pts[sim] <- n.interim[k]
        ttrial[sim] <- tobs[k]
      }
    }
  }
  
  phat <- length(which(out1==1))/nsim
  earlystop <- length(which(out1==21))/nsim
  mpts <- mean(pts)
  mtrial <- mean(ttrial)
  
  res <-list(phat, earlystop, mpts, mtrial)
  names(res) <- c("phat", "earlystop", "mpts", "mtrial")
  return(res)
}


n1init.exp <- function(init0, M, t.n, t.a, rate, FUP, nsim, alpha, beta){
  init00 <- init0 + 1
  pp <- seq(init00, M, by=5) 
  llp <-length(pp)
  qq <- seq(t.n, t.a, by=0.2) # 0.1 = three days
  ll <- length(qq)
  ahat=phat=PET <- matrix(rep(10000, ll*llp), ncol=ll)
  
  for(i in 1:llp){
    nval <- pp[i]
    n.interim = c(init0, nval)
    for(t in 1:ll){
      lamphi <- qq[t]
      out1 <- opres.exp(t.n, t.a, t.n, lam=lamphi, n.interim, rate, FUP, nsim)
      ahat[i,t] <- out1$phat
      out2 <- opres.exp(t.n, t.a, t.a, lam=lamphi, n.interim, rate, FUP, nsim)
      phat[i,t] <- 1-out2$phat 
    }
  }
  ff <- (ahat-alpha)^2+(phat-beta)^2
  ind <- which(ff == min(ff), arr.ind = TRUE)
  i1<-ind[1]
  i2<-ind[2]
  alphahat <- ahat[i1, i2]
  betahat <- phat[i1, i2]
  res <-list(betahat, alphahat)
  names(res) <- c("betahat", "alphahat")
  return(res)
}


n1init.unif <- function(init0, M, t.n, t.a, rate, FUP, nsim, 
                        para1n, para2n, para1a, para2a, alpha, beta){
  init00 <- init0 + 1
  pp <- seq(init00, M, by=5) 
  llp <-length(pp)
  qq <- seq(t.n, t.a, by=0.2) # 0.1 = three days
  ll <- length(qq)
  ahat=phat=PET <- matrix(rep(10000, ll*llp), ncol=ll)
  
  for(i in 1:llp){
    nval <- pp[i]
    n.interim = c(init0, nval)
    for(t in 1:ll){
      lamphi <- qq[t]
      out1 <- opres.unif(t.n, t.a, t.n, lam=lamphi, n.interim, rate, FUP, nsim,
                         para1=para1n, para2=para2n)
      ahat[i,t] <- out1$phat
      out2 <- opres.unif(t.n, t.a, t.a, lam=lamphi, n.interim, rate, FUP, nsim,
                         para1=para1a, para2=para2a)
      phat[i,t] <- 1-out2$phat 
    }
  }
  ff <- (ahat-alpha)^2+(phat-beta)^2
  ind <- which(ff == min(ff), arr.ind = TRUE)
  i1<-ind[1]
  i2<-ind[2]
  alphahat <- ahat[i1, i2]
  betahat <- phat[i1, i2]
  res <-list(betahat, alphahat)
  names(res) <- c("betahat", "alphahat")
  return(res)
}


n1init.wei <- function(init0, M, t.n, t.a, rate, FUP, nsim, 
                       para1n, para2n, para1a, para2a, alpha, beta){
  init00 <- init0 + 1
  pp <- seq(init00, M, by=5) 
  llp <-length(pp)
  qq <- seq(t.n, t.a, by=0.2) 
  ll <- length(qq)
  ahat=phat=PET <- matrix(rep(10000, ll*llp), ncol=ll)
  
  for(i in 1:llp){
    nval <- pp[i]
    n.interim = c(init0, nval)
    for(t in 1:ll){
      lamphi <- qq[t]
      out1 <- opres.wei(t.n, t.a, t.n, lam=lamphi, n.interim, rate, FUP, nsim,
                        para1=para1n, para2=para2n)
      ahat[i,t] <- out1$phat
      out2 <- opres.wei(t.n, t.a, t.a, lam=lamphi, n.interim, rate, FUP, nsim,
                        para1=para1a, para2=para2a)
      phat[i,t] <- 1-out2$phat 
    }
  }
  
  ff <- (ahat-alpha)^2+(phat-beta)^2
  ind <- which(ff == min(ff), arr.ind = TRUE)
  i1<-ind[1]
  i2<-ind[2]
  alphahat <- ahat[i1, i2]
  betahat <- phat[i1, i2]
  res <-list(betahat, alphahat)
  names(res) <- c("betahat", "alphahat")
  return(res)
}


METT2E <- function(alpha, beta, M, t.n, t.a, rate, FUP, nsim, 
                   nincm, lamincm, eps1, eps2, n1init, n1last, seed){
  
  if(is.null(n1init)==TRUE){
    # n1init
    n10 <- floor(M/2)
    n1res1 <- n1init.exp(n10, M, t.n, t.a, rate, FUP, nsim=1000, alpha, beta)
    if(abs(n1res1$alphahat-alpha)>eps1 & abs(n1res1$betahat-beta)>eps2) {
      n1init <- n10
    }else{
      n11 <- floor(M/4)
      n1res2 <- n1init.exp(n11, M, t.n, t.a, rate, FUP, nsim=1000, alpha, beta)
      if(abs(n1res2$alphahat-alpha)>eps1 & abs(n1res2$betahat-beta)>eps2) {
        n1init <- n11
      }else{
        n1init <- 3
      }
    }
    # print(n1init)
  }
  if(is.null(n1last)==TRUE){
    n1last <- M-1
  }
  if(is.null(seed)==TRUE){
    seed <- 840130
  }
  # use n1init
  n1 <- seq(n1init, n1last, by=1)
  n1l <- length(n1)
  n.out = lam.out <- c()
  PETn = EN =alphahat = betahat <- c()
  inderr <- rep(0, n1l)
  for(k in 1:n1l){
    cat(k, "of", n1l, ": n1 =", n1[k], "\n")
    aaa <- seed + k #840130+k
    set.seed(aaa)
    pp <- seq(n1[k] + 1, M, by=nincm) 
    llp <-length(pp)
    qq <- seq(t.n, t.a, by=lamincm) # 0.1 = three days
    ll <- length(qq)
    ahat=phat=PET <- matrix(rep(10000, ll*llp), ncol=ll)
    
    for(i in 1:llp){
      nval <- pp[i]
      for(t in 1:ll){
        lamphi <- qq[t]
        n.interim = c(n1[k], nval)
        out1 <- opres.exp(t.n, t.a, t.n, lam=lamphi, n.interim, rate, FUP, nsim)
        ahat[i,t] <- out1$phat
        out2 <- opres.exp(t.n, t.a, t.a, lam=lamphi, n.interim, rate, FUP, nsim)
        phat[i,t] <- 1-out2$phat 
        PET[i,t] <- out1$earlystop
      }
    }
    
    ff <- (ahat-alpha)^2+(phat-beta)^2
    ind <- which(ff == min(ff), arr.ind = TRUE)
    if(length(ind)>2){
      i1<-ind[1,1]
      i2<-ind[1,2]
    }else{
      i1<-ind[1]
      i2<-ind[2]
    }
    n.out[k] <- pp[i1]
    lam.out[k] <- qq[i2]
    PETn[k] <- PET[i1,i2]
    EN[k] <- n1[k]+(1-PETn[k])*(n.out[k]-n1[k])
    alphahat[k] <- ahat[i1, i2]
    betahat[k] <- phat[i1, i2]
    if (abs(ahat[i1, i2]-alpha)<eps1 & abs(phat[i1, i2]-beta)<eps2) {
      inderr[k]<-1
      cat("When n1=", n1[k], "n=", n.out[k], "and lambda=", lam.out[k],
          "Then alphahat is ", ahat[i1, i2], "and betahat is ", phat[i1, i2],
          "EN=", EN[k], "\n")
    }else{
      cat("Warning: Rule is not identified for target error rates", alpha, "and", beta, "\n")
    }
  }
  llinderr <- length(which(inderr==1))
  if(llinderr==0){
    cat("---------- Change values of n1init and n1last ----------", "\n")
    n1res= nres= lamres= enres= petres= ares= bres <- NA
    
  }else{
    # extract info
    EN1 <- EN[which(inderr==1)]
    enres <- min(EN1)
    n11 <- n1[which(inderr==1)]
    n1res <- n11[which(EN1==min(EN1))]
    nres1 <- n.out[which(inderr==1)]
    nres <- nres1[which(EN1==min(EN1))]
    lamres1 <- lam.out[which(inderr==1)]
    lamres <- lamres1[which(EN1==min(EN1))]
    PETn1 <- PETn[which(inderr==1)]
    petres <- PETn1[which(EN1==min(EN1))]
    ares1 <- alphahat[which(inderr==1)]
    ares <- ares1[which(EN1==min(EN1))]
    bres1 <- betahat[which(inderr==1)]
    bres <- bres1[which(EN1==min(EN1))]
  }
  
  res <- list(n1res, nres, lamres, enres, petres, ares, bres)
  names(res) <- c("n1", "n", "lambda", "EN", "PET0", "alphahat", "betahat") 
  return(res) 
}


METT2U <- function(alpha, beta, M, t.n, t.a, rate, FUP, nsim, 
                   nincm, lamincm, eps1, eps2, n1init, n1last, seed){
  
  if(is.null(n1init)==TRUE){
    n10 <- floor(M/2)
    n1res1 <- n1init.unif(n10, M,
                          t.n, t.a, rate, FUP, nsim=1000,
                          para1n=0, para2n=2*t.n, para1a=0, para2a=2*t.a, alpha, beta)
    if(abs(n1res1$alphahat-alpha)>eps1 & abs(n1res1$betahat-beta)>eps2) {
      n1init <- n10
    }else{
      n11 <- floor(M/4)
      n1res2 <- n1init.unif(n11, M,
                            t.n, t.a, rate, FUP, nsim=1000,
                            para1n=0, para2n=2*t.n, para1a=0, para2a=2*t.a, alpha, beta)
      if(abs(n1res2$alphahat-alpha)>eps1 & abs(n1res2$betahat-beta)>eps2) {
        n1init <- n11
      }else{
        n1init <- 3
      }
    }
    # print(n1init)
  }
  if(is.null(n1last)==TRUE){
    n1last <- M-1
  }
  if(is.null(seed)==TRUE){
    seed <- 840130
  }
  # use n1init
  n1 <- seq(n1init, n1last, by=1)
  n1l <- length(n1)
  n.out = lam.out <- c()
  PETn = EN =alphahat = betahat <- c()
  inderr <- rep(0, n1l)
  for(k in 1:n1l){
    cat(k, "of", n1l, ": n1 =", n1[k], "\n")
    aaa <- seed + k #840130+k
    set.seed(aaa)
    pp <- seq(n1[k] + 1, M, by=nincm) 
    llp <-length(pp)
    qq <- seq(t.n, t.a, by=lamincm) # 0.1 = three days
    ll <- length(qq)
    ahat=phat=PET <- matrix(rep(10000, ll*llp), ncol=ll)
    
    for(i in 1:llp){
      nval <- pp[i]
      for(t in 1:ll){
        lamphi <- qq[t]
        n.interim = c(n1[k], nval)
        out1 <- opres.unif(t.n, t.a, t.n, lam=lamphi, n.interim, rate, FUP, nsim,
                           para1=0, para2=2*t.n)
        ahat[i,t] <- out1$phat
        out2 <- opres.unif(t.n, t.a, t.a, lam=lamphi, n.interim, rate, FUP, nsim,
                           para1=0, para2=2*t.a)
        phat[i,t] <- 1-out2$phat 
        PET[i,t] <- out1$earlystop
      }
    }
    
    ff <- (ahat-alpha)^2+(phat-beta)^2
    ind <- which(ff == min(ff), arr.ind = TRUE)
    if(length(ind)>2){
      i1<-ind[1,1]
      i2<-ind[1,2]
    }else{
      i1<-ind[1]
      i2<-ind[2]
    }
    n.out[k] <- pp[i1]
    lam.out[k] <- qq[i2]
    PETn[k] <- PET[i1,i2]
    EN[k] <- n1[k]+(1-PETn[k])*(n.out[k]-n1[k])
    alphahat[k] <- ahat[i1, i2]
    betahat[k] <- phat[i1, i2]
    if (abs(ahat[i1, i2]-alpha)<eps1 & abs(phat[i1, i2]-beta)<eps2) {
      inderr[k]<-1
      cat("When n1=", n1[k], "n=", n.out[k], "and lambda=", lam.out[k],
          "Then alphahat is ", ahat[i1, i2], "and betahat is ", phat[i1, i2],
          "EN=", EN[k], "\n")
    }else{
      cat("Warning: Rule is not identified for target error rates", alpha, "and", beta, "\n")
    }
  }
  llinderr <- length(which(inderr==1))
  if(llinderr==0){
    cat("---------- Change values of n1init and n1last ----------", "\n")
    n1res= nres= lamres= enres= petres= ares= bres <- NA
    
  }else{
    # extract info
    EN1 <- EN[which(inderr==1)]
    enres <- min(EN1)
    n11 <- n1[which(inderr==1)]
    n1res <- n11[which(EN1==min(EN1))]
    nres1 <- n.out[which(inderr==1)]
    nres <- nres1[which(EN1==min(EN1))]
    lamres1 <- lam.out[which(inderr==1)]
    lamres <- lamres1[which(EN1==min(EN1))]
    PETn1 <- PETn[which(inderr==1)]
    petres <- PETn1[which(EN1==min(EN1))]
    ares1 <- alphahat[which(inderr==1)]
    ares <- ares1[which(EN1==min(EN1))]
    bres1 <- betahat[which(inderr==1)]
    bres <- bres1[which(EN1==min(EN1))]
  }
  
  res <- list(n1res, nres, lamres, enres, petres, ares, bres)
  names(res) <- c("n1", "n", "lambda", "EN", "PET0", "alphahat", "betahat") 
  return(res) 
}


METT2W <- function(alpha, beta, M, t.n, t.a, rate, FUP, nsim, 
                   nincm, lamincm, eps1, eps2, n1init, n1last, seed){
  
  kkk <- 2
  if(is.null(n1init)==TRUE){
    n10 <- floor(M/2)
    n1res1 <- n1init.wei(n10, M,
                         t.n, t.a, rate, FUP, nsim=1000,
                         para1n=kkk, para2n=t.n/(log(2))^(1/kkk), 
                         para1a=kkk, para2a=t.a/(log(2))^(1/kkk), alpha, beta)
    if(abs(n1res1$alphahat-alpha)>eps1 & abs(n1res1$betahat-beta)>eps2) {
      n1init <- n10
    }else{
      n11 <- floor(M/4)
      n1res2 <- n1init.wei(n11, M,
                           t.n, t.a, rate, FUP, nsim=1000,
                           para1n=kkk, para2n=t.n/(log(2))^(1/kkk), 
                           para1a=1, para2a=t.a/(log(2))^(1/kkk), alpha, beta)
      if(abs(n1res2$alphahat-alpha)>eps1 & abs(n1res2$betahat-beta)>eps2) {
        n1init <- n11
      }else{
        n1init <- 3
      }
    }
    # print(n1init)
  }
  if(is.null(n1last)==TRUE){
    n1last <- M-1
  }
  if(is.null(seed)==TRUE){
    seed <- 840130
  }
  # use n1init
  n1 <- seq(n1init, n1last, by=1)
  n1l <- length(n1)
  n.out = lam.out <- c()
  PETn = EN =alphahat = betahat <- c()
  inderr <- rep(0, n1l)
  for(k in 1:n1l){
    cat(k, "of", n1l, ": n1 =", n1[k], "\n")
    aaa <- seed + k #840130+k
    set.seed(aaa)
    pp <- seq(n1[k] + 1, M, by=nincm) 
    llp <-length(pp)
    qq <- seq(t.n, t.a, by=lamincm) # 0.1 = three days
    ll <- length(qq)
    ahat=phat=PET <- matrix(rep(10000, ll*llp), ncol=ll)
    
    for(i in 1:llp){
      nval <- pp[i]
      for(t in 1:ll){
        lamphi <- qq[t]
        n.interim = c(n1[k], nval)
        out1 <- opres.wei(t.n, t.a, t.n, lam=lamphi, n.interim, rate, FUP, nsim,
                          para1=kkk, para2=t.n/(log(2))^(1/kkk))
        ahat[i,t] <- out1$phat
        out2 <- opres.wei(t.n, t.a, t.a, lam=lamphi, n.interim, rate, FUP, nsim,
                          para1=kkk, para2=t.a/(log(2))^(1/kkk))
        phat[i,t] <- 1-out2$phat 
        PET[i,t] <- out1$earlystop
      }
    }
    
    ff <- (ahat-alpha)^2+(phat-beta)^2
    ind <- which(ff == min(ff), arr.ind = TRUE)
    if(length(ind)>2){
      i1<-ind[1,1]
      i2<-ind[1,2]
    }else{
      i1<-ind[1]
      i2<-ind[2]
    }
    n.out[k] <- pp[i1]
    lam.out[k] <- qq[i2]
    PETn[k] <- PET[i1,i2]
    EN[k] <- n1[k]+(1-PETn[k])*(n.out[k]-n1[k])
    alphahat[k] <- ahat[i1, i2]
    betahat[k] <- phat[i1, i2]
    if (abs(ahat[i1, i2]-alpha)<eps1 & abs(phat[i1, i2]-beta)<eps2) {
      inderr[k]<-1
      cat("When n1=", n1[k], "n=", n.out[k], "and lambda=", lam.out[k],
          "Then alphahat is ", ahat[i1, i2], "and betahat is ", phat[i1, i2],
          "EN=", EN[k], "\n")
    }else{
      cat("Warning: Rule is not identified for target error rates", alpha, "and", beta, "\n")
    }
    
  }
  llinderr <- length(which(inderr==1))
  if(llinderr==0){
    cat("---------- Change values of n1init and n1last ----------", "\n")
    n1res= nres= lamres= enres= petres= ares= bres <- NA
    
  }else{
    # extract info
    EN1 <- EN[which(inderr==1)]
    enres <- min(EN1)
    n11 <- n1[which(inderr==1)]
    n1res <- n11[which(EN1==min(EN1))]
    nres1 <- n.out[which(inderr==1)]
    nres <- nres1[which(EN1==min(EN1))]
    lamres1 <- lam.out[which(inderr==1)]
    lamres <- lamres1[which(EN1==min(EN1))]
    PETn1 <- PETn[which(inderr==1)]
    petres <- PETn1[which(EN1==min(EN1))]
    ares1 <- alphahat[which(inderr==1)]
    ares <- ares1[which(EN1==min(EN1))]
    bres1 <- betahat[which(inderr==1)]
    bres <- bres1[which(EN1==min(EN1))]
  }
  
  res <- list(n1res, nres, lamres, enres, petres, ares, bres)
  names(res) <- c("n1", "n", "lambda", "EN", "PET0", "alphahat", "betahat") 
  return(res) 
}

