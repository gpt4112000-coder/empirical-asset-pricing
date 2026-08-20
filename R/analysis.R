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

# --- Robustness checks, mirroring robustness.py ---

# VIF via 1/(1-R^2) of each regressor on the other regressors -- avoids a
# car::vif dependency (car pulls in a heavy dependency tree) while giving
# the identical statistic.
vif_manual <- function(X) {
  vifs <- c()
  for (col in colnames(X)) {
    other <- setdiff(colnames(X), col)
    r2 <- summary(lm(as.formula(paste(col, "~", paste(other, collapse="+"))), data=X))$r.squared
    vifs[col] <- 1 / (1 - r2)
  }
  vifs
}
factor_cols <- df[, c("Mkt-RF", "SMB", "HML")]
colnames(factor_cols) <- c("Mkt.RF", "SMB", "HML")  # backtick-safe names for formula use
vif_result <- vif_manual(factor_cols)
cat("\n=== VIF (factor multicollinearity) ===\n")
print(vif_result)

# Subperiod split: first half vs second half, same HAC(5) spec as full sample
mid <- floor(nrow(df) / 2)
first_half <- df[1:mid, ]
second_half <- df[(mid+1):nrow(df), ]
fit_first <- lm(excess ~ `Mkt-RF` + SMB + HML, data=first_half)
fit_second <- lm(excess ~ `Mkt-RF` + SMB + HML, data=second_half)
cat("\n=== Subperiod split: first half ===\n")
print(coeftest(fit_first, vcov=vcovHAC(fit_first, lag=5, type="HC0")))
cat("\n=== Subperiod split: second half ===\n")
print(coeftest(fit_second, vcov=vcovHAC(fit_second, lag=5, type="HC0")))

# Rolling 252-day betas
window <- 252
roll_rows <- list()
if (nrow(df) > window) {
  for (end in window:nrow(df)) {
    seg <- df[(end - window + 1):end, ]
    seg_fit <- tryCatch(lm(excess ~ `Mkt-RF` + SMB + HML, data=seg), error=function(e) NULL)
    if (!is.null(seg_fit)) {
      cf <- coef(seg_fit)
      roll_rows[[length(roll_rows)+1]] <- data.frame(
        Date=seg$Date[nrow(seg)], alpha=cf[1], beta_mkt=cf[2], beta_smb=cf[3], beta_hml=cf[4]
      )
    }
  }
  rolling_df <- do.call(rbind, roll_rows)
  write.csv(rolling_df, "results/ff3_R_rolling_betas.csv", row.names=FALSE)
  cat(sprintf("\n[R] wrote results/ff3_R_rolling_betas.csv (%d rolling windows)\n", nrow(rolling_df)))
}
cat("[R] robustness checks complete\n")
