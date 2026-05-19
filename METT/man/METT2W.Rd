\name{METT2W}
\alias{METT2W}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Optimal Two-Stage Design based on Median Event Time Test for Weibull survivals.
}
\description{
It numerically searches the optimal decision rule of two-stage design based on median event time test when survivals are assumed to follow Weibull distribution. 
}
\usage{
METT2W(alpha, beta, M, t.n, t.a, rate, FUP, nsim, nincm, lamincm, eps1, eps2, n1init, n1last, seed)
}
\arguments{
  \item{alpha}{
Target type I error rate
}
  \item{beta}{
Target type II error rate
}
  \item{M}{
Maximum sample size for the study
}
  \item{t.n}{
Null median event time
}
  \item{t.a}{
Alternative median event time
}
  \item{rate}{
Patient accrual rate per month
}
  \item{FUP}{
Follow-up time (months)
}
  \item{nsim}{
Number of simulations
}
  \item{nincm}{
Increment of the sequence for sample size to be searched
}
  \item{lamincm}{
Increment of the sequence for lambda to be searched
}
  \item{eps1}{
Maximum acceptable difference between the estimated type I error rate from the trial design and the target error rate (alpha), epsilon1
}
  \item{eps2}{
Maximum acceptable difference between the estimated type I error rate from the trial design and the target error rate (beta), epsilon2
}
  \item{n1init}{
Initial value of n1 to be searched
}
  \item{n1last}{
Last value of n1 to be searched
}
  \item{seed}{
seed number 
}
}
\value{
\item{n1}{The number of patients for the first stage}
\item{n}{Total sample size}
\item{lambda}{Threshold of the median event time test}
\item{EN}{Expected sample size foe the two-stage design}
\item{PET0}{Probability of early termination after the first stage under the null hypothesis}
\item{alphahat}{Estimated type I error rate}
\item{betahat}{Estimated type II error rate}
}
\references{
Park, Y. (2021+) Optimal Two-Stage Design of Single arm Phase II Clinical Trials based on Median Event Time Test.
}
\section{Warning }{
Depending on values of nincm and lamincm, it takes a lot of time to seach the optimal solution. It is recommended to adjust values of n1init and n1last and run multiple jobs.
}