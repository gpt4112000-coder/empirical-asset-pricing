# R version of FF3 replication — STATA/MATLAB-equivalent workflow in R
# Run: Rscript R/analysis.R
# Falls back to synthetic if data missing.

# Check packages
if (!require("sandwich")) install.packages("sandwich", repos="https://cloud.r-project.org")
if (!require("lmtest")) install.packages("lmtest", repos="https://cloud.r-project.org")
library(sandwich)
library(lmtest)

ROOT <- if (file.exists("data/ff3.csv")) "." else "."
FF <- file.path(ROOT, "data/ff3.csv")
PX <- "../app-0001-nk-securities-quant-researcher/backtested-strategy-engine/data/prices.csv"
# try two locations
if (!file.exists(PX)) PX <- "../app-0001-nk-securities-quant-researcher/project/data/prices.csv"
if (!file.exists(PX)) PX <- file.path(ROOT, "../app-0001-nk-securities-quant-researcher/backtested-strategy-engine/data/prices.csv")
if (!file.exists(PX)) PX <- file.path(ROOT, "../app-0001-nk-securities-quant-researcher/project/data/prices.csv")
if (!file.exists(PX)) {
  # fallback search
  candidates <- list.files("../..", pattern="prices.csv", recursive=TRUE, full.names=TRUE)
  if (length(candidates) > 0) PX <- candidates[1]
}

cat(sprintf("[R] FF: %s exists=%s\n", FF, file.exists(FF)))
cat(sprintf("[R] PX: %s exists=%s\n", PX, file.exists(PX)))

ff <- read.csv(FF)
ff$Date <- as.Date(ff$Date)
px <- read.csv(PX)
px$Date <- as.Date(px$Date)

# equal-weight portfolio return per day
# Use aggregate
px <- px[order(px$Ticker, px$Date),]
# compute ret per ticker
px$ret <- ave(px$Close, px$Ticker, FUN=function(x) c(NA, diff(x)/x[-length(x)]))
daily <- aggregate(ret ~ Date, data=px, FUN=mean, na.rm=TRUE)
names(daily)[2] <- "ret"
daily <- daily[!is.na(daily$ret),]

df <- merge(daily, ff, by="Date")
df$excess <- df$ret - df$RF
cat(sprintf("[R] merged %d rows %s -> %s\n", nrow(df), min(df$Date), max(df$Date)))

# OLS
fit <- lm(excess ~ `Mkt-RF` + SMB + HML, data=df)
cat("\n=== OLS Summary ===\n")
print(summary(fit))

# Newey-West (HAC, 5 lags) — sandwich equivalent of STATA newey
cat("\n=== Newey-West (sandwich, lag=5) ===\n")
nw_vcov <- vcovHAC(fit, lag=5, type="HC0")
print(coeftest(fit, vcov=nw_vcov))

# Save coefficients
res <- data.frame(
  coef = coef(fit),
  se_ols = summary(fit)$coefficients[,2],
  se_nw = sqrt(diag(nw_vcov)),
  p_nw = coeftest(fit, vcov=nw_vcov)[,4]
)
print(res)
dir.create("results", showWarnings=FALSE)
write.csv(res, "results/ff3_R_coefficients.csv", row.names=TRUE)
cat("[R] saved results/ff3_R_coefficients.csv\n")

# Diagnostics plot
png("results/ff3_R_diagnostics.png", width=1000, height=400)
par(mfrow=c(1,2))
plot(fitted(fit), df$excess, pch=16, cex=0.4, col=rgb(0,0,0,0.3),
     xlab="Fitted excess", ylab="Actual excess",
     main=sprintf("FF3 R²=%.3f n=%d", summary(fit)$r.squared, nrow(df)))
abline(a=0,b=1,col="red",lty=2)
plot(df$Date, resid(fit), type="l", col="grey30", lwd=0.5,
     main="Residuals over time", ylab="Residual", xlab="Date")
dev.off()
cat("[R] saved diagnostics\n")
