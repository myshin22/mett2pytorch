\name{METT-package}
\alias{METT-package}
\alias{METT}
\docType{package}

\title{
Optimal Two-Stage Design of Single arm Phase II Clinical Trials based on Median Event Time Test
}

\description{
\packageDescription{METT}
}

\details{

The DESCRIPTION file:
\packageDESCRIPTION{METT}
\packageIndices{METT}

There are three main functions. Function \code{METT2E} provides a decision rule of optimal two-stage design assuming exponential survivals. Function \code{METT2U} provides a decision rule of optimal two-stage design assuming uniform survivals. Function \code{METT2W} provides a decision rule of optimal two-stage design assuming weibull survivals.
}

\author{
\packageAuthor{METT}

Maintainer: \packageMaintainer{METT}
}
\references{
Park, Y. (2021) Optimal Two-Stage Design of Single arm Phase II Clinical Trials based on Median Event Time Test. PLoS One. 16(2): e0246448.
}
\keyword{Clinical trials, median event time, optimal design, single arm, time-to-event}

\examples{
% library(survival)
% library(METT)

% out1 <- METT2E(alpha=0.05, beta=0.2, M=21, t.n=3, t.a=7, rate=1.04, FUP=24,
%                nsim=1000, nincm=1, lamincm=0.1, eps1=0.02, eps2=0.025,
%                n1init=10, n1last=15, seed=NULL)

% out2 <- METT2U(alpha=0.05, beta=0.2, M=21, t.n=3, t.a=7, rate=1.04, FUP=24,
%                nsim=1000, nincm=1, lamincm=0.1, eps1=0.02, eps2=0.025,
%                n1init=5, n1last=10, seed=NULL)
 
% out3 <- METT2W(alpha=0.05, beta=0.2, M=21, t.n=3, t.a=7, rate=1.04, FUP=24,
%                nsim=1000, nincm=1, lamincm=0.1, eps1=0.02, eps2=0.025,
%                n1init=5, n1last=10, seed=NULL)
}
