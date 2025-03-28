## HOW TO INSTALL T-LOCOH IN 2024

## The last real update to tlocoh was in 2013. Since then, a couple of the dependent packages,
## including rgdal, rgeos, and gcplib, have been retired. This means you won't be able to 
## install tlocoh on a 'fresh' installation of R.

## The package however still works with an older version of R. The instructions below
## describe how to install an old version R (don't worry, you don't have to uninstall your
## current version of R. R versions are installed in separate directories, and RStudio allows you to
## select which version to load). 

## You will also need to install older versions of the dependent packages (which is not that hard, 
## thanks to groundhog), as well as RTools 3.5 (because RTools 4 only works with R 4.x).

## 1) Install an older version of R (say 3.6.3)
##    Download an older version of R from 
##    https://cran.r-project.org/bin/windows/base/old/
##    Install it.

## 2. Install RTools 3.5
##    https://cran.r-project.org/bin/windows/Rtools/Rtools35.exe
##    Select the option to add it to the path
##    Why do I need RTools? Because the gcplib package needs compiling
##    More info: https://groundhogr.com/help-with-r-tools-only-for-windows-users/

## 3. In RStudio, select the older version you just installed:
##      Tools > Global Options > General > R Version
##    Restart RStudio

## Verify RTools 3.5 is installed and on the path:
Sys.which("make")

## Install old versions of the tlocoh dependent packages:

req_pkgs <- c("sp", "FNN", "pbapply", "rgeos", "rgdal", "move", "png", "raster", "XML", "gpclib")

back_date <- "2020-01-01"

library("groundhog")

for (old_pkg in req_pkgs) {
  cat("Installing old version of ", old_pkg, "\n")
  groundhog.library(old_pkg, back_date)
}

## Install tlocoh (minus the dependencies, which we just installed above)
install.packages("tlocoh", dependencies=FALSE, repos="http://R-Forge.R-project.org")

## Run a sample script to see if it worked.
## The following script is taken from the Tlocoh Tutorial and Users Guide
## https://tlocoh.r-forge.r-project.org/tlocoh_tutorial_2014-08-17.pdf

library(tlocoh)

data(toni)
class(toni)
head(toni)
plot(toni[ , c("long","lat")], pch=20)
require(sp)
require(rgdal)
toni.sp.latlong <- SpatialPoints(toni[ , c("long","lat")],
                                 proj4string=CRS("+proj=longlat +ellps=WGS84"))
toni.sp.utm <- spTransform(toni.sp.latlong, CRS("+proj=utm +south +zone=36 +ellps=WGS84"))
toni.mat.utm <- coordinates(toni.sp.utm)
head(toni.mat.utm)
colnames(toni.mat.utm) <- c("x","y")
head(toni.mat.utm)
class(toni$timestamp.utc)
head(as.character(toni$timestamp.utc))
toni.gmt <- as.POSIXct(toni$timestamp.utc, tz="UTC")
toni.gmt[1:3]
local.tz <- "Africa/Johannesburg"
toni.localtime <- as.POSIXct(format(toni.gmt, tz=local.tz), tz=local.tz)
toni.localtime[1:3]
toni.lxy <- xyt.lxy(xy=toni.mat.utm, dt=toni.localtime, id="toni",
                    proj4string=CRS("+proj=utm +south +zone=36 +ellps=WGS84"))
summary(toni.lxy)
plot(toni.lxy)
hist(toni.lxy)
lxy.plot.freq(toni.lxy, deltat.by.date=T)
lxy.plot.freq(toni.lxy, cp=T)
toni.lxy <- lxy.thin.bursts(toni.lxy, thresh=0.2)
toni.lxy <- lxy.ptsh.add(toni.lxy)
lxy.plot.sfinder(toni.lxy)
lxy.plot.sfinder(toni.lxy, delta.t=3600*c(12,24,36,48,54,60))
toni.lxy <- lxy.nn.add(toni.lxy, s=c(0.0003, 0.003, 0.03, 0.3), k=25)
summary(toni.lxy)
lxy.plot.mtdr(toni.lxy, k=10)
lxy.plot.tspan(toni.lxy, k=10)
toni.lhs <- lxy.lhs(toni.lxy, k=3*2:8, s=0.003)
summary(toni.lhs, compact=T)
toni.lhs <- lhs.iso.add(toni.lhs)
plot(toni.lhs, iso=T)
plot(toni.lhs, iso=T, k=15, allpts=T, cex.allpts=0.1, col.allpts="gray30")
lhs.plot.isoarea(toni.lhs)
lhs.plot.isoear(toni.lhs)
toni.lhs.k15 <- lhs.select(toni.lhs, k=15)

## a-method
summary(toni.lxy)
toni.lxy <- lxy.nn.add(toni.lxy, s=0.003, a=auto.a(nnn=15, ptp=0.98))
summary(toni.lxy)
toni.lxy <- lxy.nn.add(toni.lxy, s=0.003, a=15000)
toni.lhs.amixed <- lxy.lhs(toni.lxy, s=0.003, a=4:15*1000, iso.add=T)
lhs.plot.isoarea(toni.lhs.amixed)
lhs.plot.isoear(toni.lhs.amixed)

## Hull metrics
toni.lhs.k15 <- lhs.ellipses.add(toni.lhs.k15)  ## takes ~2 minutes
summary(toni.lhs.k15)
plot(toni.lhs.k15, hulls=T, ellipses=T, allpts=T, nn=T, ptid="auto")
toni.lhs.k15 <- lhs.visit.add(toni.lhs.k15, ivg=3600*12)

toni.lhs.k15 <- lhs.iso.add(toni.lhs.k15, sort.metric="ecc")
plot(toni.lhs.k15, iso=T, iso.sort.metric="ecc")
hist(toni.lhs.k15, metric="nsv")
plot(toni.lhs.k15, hpp=T, hpp.classify="nsv", ivg=3600*12, col.ramp="rainbow")


