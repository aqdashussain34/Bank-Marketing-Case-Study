# =========================
# Bank Marketing Case Study – Full Script (hardened factor alignment + level seeding)
# =========================

# ---- Packages ----
library(tidyverse)
library(caret)
library(ROSE)
library(MASS)
library(pROC)
library(dplyr)
library(forcats)
library(broom)

set.seed(42)

# ---- Load ----
setwd("/Users/achyutparmarthi/Desktop/Data App/week 3/")
bank <- read.csv("bank-additional-full.csv", sep = ";") %>%
  dplyr::mutate(y = factor(y, levels = c("no","yes"))) %>%
  dplyr::select(-duration)  # duration excluded per assignment

# ---- Part A: Data Understanding & Preparation ----
cat("\n=== Part A: Data Understanding ===\n")
cat("Rows:", nrow(bank), " | Cols:", ncol(bank), "\n")

is_cat <- sapply(bank, is.factor)
cats   <- names(bank)[is_cat]
nums   <- names(bank)[!is_cat & names(bank) != "y"]
cat("\nCategorical variables (", length(cats), "):\n", paste(cats, collapse = ", "), "\n", sep = "")
cat("\nNumeric variables (", length(nums), "):\n", paste(nums, collapse = ", "), "\n", sep = "")

class_tbl  <- table(bank$y)
class_prop <- prop.table(class_tbl)
cat("\nClass balance:\n"); print(class_tbl); print(round(class_prop, 4))

# Feature engineering
bank <- bank %>%
  dplyr::mutate(
    age_group = dplyr::case_when(
      age <= 25 ~ "17-25",
      age <= 35 ~ "26-35",
      age <= 45 ~ "36-45",
      age <= 55 ~ "46-55",
      age <= 65 ~ "56-65",
      TRUE      ~ "66+"
    ),
    pdays_group = dplyr::case_when(
      pdays == 999 ~ "Not_Contacted",
      pdays <= 90  ~ "0-90",
      pdays <= 180 ~ "91-180",
      pdays <= 270 ~ "181-270",
      pdays <= 360 ~ "271-360",
      TRUE         ~ "360+"
    )
  )

cat("\nEngineered variables added: age_group, pdays_group; excluded: duration\n")

# ---- Split ----
idx   <- caret::createDataPartition(bank$y, p = 0.7, list = FALSE)
train <- bank[idx, ]
test  <- bank[-idx, ]

# ---- Explicit typing: which predictors are categorical vs numeric ----
cat_vars <- c("age_group","pdays_group",
              "job","marital","education","default","housing","loan",
              "contact","month","day_of_week","poutcome")
num_vars <- c("campaign","previous",
              "emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed")

# Keep only columns that exist
cat_vars <- intersect(cat_vars, names(train))
num_vars <- intersect(num_vars, names(train))

# Coerce types in both train & test BEFORE ROSE
to_factor <- function(df, cols) {
  for (c in cols) df[[c]] <- factor(as.character(df[[c]]))
  df
}
to_numeric <- function(df, cols) {
  for (c in cols) df[[c]] <- suppressWarnings(as.numeric(df[[c]]))
  df
}
train <- to_factor(train, cat_vars); test <- to_factor(test, cat_vars)
train <- to_numeric(train, num_vars);  test <- to_numeric(test, num_vars)

# ---- Part B: Modeling ----
predictors <- c(cat_vars, num_vars)
form <- as.formula(paste("y ~", paste(predictors, collapse = " + ")))

# ROSE resampling on the typed data
train_bal <- ROSE::ROSE(form, data = train, seed = 42)$data

# Re-assert types post-ROSE (ROSE can sometimes coerce)
train_bal <- to_factor(train_bal, cat_vars)
train_bal <- to_numeric(train_bal, num_vars)

# ---- Robust factor level alignment for ALL categorical predictors ----
align_levels <- function(train_df, test_df, cols) {
  for (col in cols) {
    # ensure "Other" is a possible training level
    train_df[[col]] <- forcats::fct_expand(train_df[[col]], "Other")
    keep_lvls <- levels(train_df[[col]])
    
    # map test unseen -> "Other", coerce to same levels/order
    test_df[[col]] <- forcats::fct_other(test_df[[col]], keep = keep_lvls, other_level = "Other")
    test_df[[col]] <- factor(test_df[[col]], levels = keep_lvls)
  }
  list(train = train_df, test = test_df)
}
tmp <- align_levels(train_bal, test, cat_vars)
train_bal <- tmp$train
test      <- tmp$test
rm(tmp)

# ---- NEW: Seed any test-only levels into training (e.g., "Other") ----
seed_missing_levels <- function(train_df, test_df, cols) {
  for (col in cols) {
    if (!is.factor(train_df[[col]]) || !is.factor(test_df[[col]])) next
    # levels present in test, absent in training counts
    test_lvls_used <- names(which(table(test_df[[col]]) > 0))
    for (lv in test_lvls_used) {
      if (!(lv %in% levels(train_df[[col]]))) next  # skip truly unseen in both; shouldn't happen
      # if level exists in levels but zero rows have it, seed one row
      if (sum(train_df[[col]] == lv, na.rm = TRUE) == 0) {
        idx_any <- which(!is.na(train_df[[col]]))[1]
        if (length(idx_any) == 1 && !is.na(idx_any)) {
          train_df[[col]][idx_any] <- lv
        }
      }
    }
  }
  train_df
}
train_bal <- seed_missing_levels(train_bal, test, cat_vars)

# ------------------
# Model 1: LOGISTIC
# ------------------
log_model <- glm(form, data = train_bal, family = binomial)
log_probs <- predict(log_model, newdata = test, type = "response")

ok_log       <- !is.na(log_probs) & !is.na(test$y)
log_probs_ok <- log_probs[ok_log]
y_ok         <- test$y[ok_log]

roc_log <- pROC::roc(y_ok, log_probs_ok, levels = c("no","yes"), quiet = TRUE)
auc_log <- as.numeric(pROC::auc(roc_log))
thr_you <- as.numeric(pROC::coords(roc_log, "best", best.method = "youden", ret = "threshold"))
if (length(thr_you) > 1) thr_you <- mean(thr_you)

log_pred_you <- factor(ifelse(log_probs_ok >= thr_you, "yes", "no"), levels = c("no","yes"))
log_cm_you   <- caret::confusionMatrix(log_pred_you, y_ok, positive = "yes")

# ---- Sensitivity-tuned threshold (maximize sensitivity, tie-break by specificity) ----
coords_all <- as.data.frame(pROC::coords(
  roc_log, x = "all",
  ret = c("threshold","sensitivity","specificity"),
  transpose = FALSE
))

best_sens <- max(coords_all$sensitivity, na.rm = TRUE)
cand <- coords_all[coords_all$sensitivity == best_sens, , drop = FALSE]
thr_sens <- cand$threshold[which.max(cand$specificity)]

log_pred_sens <- factor(ifelse(log_probs_ok >= thr_sens, "yes", "no"),
                        levels = c("no","yes"))
log_cm_sens <- caret::confusionMatrix(log_pred_sens, y_ok, positive = "yes")

cat("\nLogistic tuned for sensitivity (max sensitivity):\n")
cat(sprintf("Chosen Thr: %.3f | Sens: %.3f | Spec: %.3f\n",
            thr_sens,
            cand$sensitivity[which.max(cand$specificity)],
            cand$specificity[which.max(cand$specificity)]))
print(log_cm_sens$byClass[c("Sensitivity","Specificity","Balanced Accuracy")])


# ------------------
# Model 2: LDA (robust dummying)
# ------------------
dv <- caret::dummyVars(~ . - y, data = train_bal)  # dummy predictors only
X_train <- as.data.frame(predict(dv, newdata = train_bal))
X_test  <- as.data.frame(predict(dv, newdata = test))

# Drop NZV columns
nzv_idx <- caret::nearZeroVar(X_train)
if (length(nzv_idx) > 0) {
  X_train <- X_train[ , -nzv_idx, drop = FALSE]
  X_test  <- X_test[  , -nzv_idx, drop = FALSE]
}

# Drop columns constant within any class in training
const_within_class <- sapply(colnames(X_train), function(cn) {
  any(tapply(X_train[[cn]], train_bal$y, function(v) length(unique(v)) <= 1))
})
if (any(const_within_class)) {
  X_train <- X_train[ , !const_within_class, drop = FALSE]
  X_test  <- X_test[  , !const_within_class, drop = FALSE]
}

row_ok_train <- stats::complete.cases(X_train)
X_train2 <- X_train[row_ok_train, , drop = FALSE]
y_train2 <- droplevels(train_bal$y[row_ok_train])

row_ok_test <- stats::complete.cases(X_test) & !is.na(test$y)
X_test2 <- X_test[row_ok_test, , drop = FALSE]
y_test2 <- droplevels(test$y[row_ok_test])

lda_model <- MASS::lda(x = X_train2, grouping = y_train2)
lda_out   <- predict(lda_model, newdata = X_test2)
lda_probs <- lda_out$posterior[, "yes"]

roc_lda <- pROC::roc(y_test2, lda_probs, levels = c("no","yes"), quiet = TRUE)
auc_lda <- as.numeric(pROC::auc(roc_lda))
thr_lda_you <- as.numeric(pROC::coords(roc_lda, "best", best.method = "youden", ret = "threshold"))
if (length(thr_lda_you) > 1) thr_lda_you <- mean(thr_lda_you)

lda_pred_you <- factor(ifelse(lda_probs >= thr_lda_you, "yes", "no"), levels = c("no","yes"))
lda_cm_you   <- caret::confusionMatrix(lda_pred_you, y_test2, positive = "yes")

# ---- Part B: Comparison output ----
cat("\n=== Part B: Modeling – Summary ===\n")
cat(sprintf("Logistic AUC: %.4f | Youden Thr: %.3f\n", auc_log, thr_you))
print(log_cm_you$byClass[c("Sensitivity","Specificity","Balanced Accuracy")])

# Use the 'cand' row we computed above (max sensitivity; tie-broken by max specificity)
achieved_sens <- cand$sensitivity[which.max(cand$specificity)]
achieved_spec <- cand$specificity[which.max(cand$specificity)]

cat("\nLogistic tuned for sensitivity (max sensitivity):\n")
cat(sprintf("Chosen Thr: %.3f | Sens: %.3f | Spec: %.3f\n",
            thr_sens, achieved_sens, achieved_spec))
print(log_cm_sens$byClass[c("Sensitivity","Specificity","Balanced Accuracy")])

cat(sprintf("\nLDA AUC: %.4f | Youden Thr: %.3f\n", auc_lda, thr_lda_you))
print(lda_cm_you$byClass[c("Sensitivity","Specificity","Balanced Accuracy")])


# ==========================================================
# Extra Step: Evaluate on a balanced test set
# ==========================================================

# Create a balanced test set using ROSE
test_bal <- ROSE::ROSE(y ~ ., data = test, seed = 42)$data

# Logistic model predictions on balanced test
log_probs_bal <- predict(log_model, newdata = test_bal, type = "response")
roc_log_bal   <- pROC::roc(test_bal$y, log_probs_bal, levels = c("no","yes"), quiet = TRUE)
auc_log_bal   <- as.numeric(pROC::auc(roc_log_bal))
thr_bal       <- as.numeric(pROC::coords(roc_log_bal, "best", best.method = "youden", ret = "threshold"))
if (length(thr_bal) > 1) thr_bal <- mean(thr_bal)

log_pred_bal <- factor(ifelse(log_probs_bal >= thr_bal, "yes", "no"), levels = c("no","yes"))
log_cm_bal   <- caret::confusionMatrix(log_pred_bal, test_bal$y, positive = "yes")

cat("\n=== Logistic Regression (Balanced Test) ===\n")
cat(sprintf("AUC: %.4f | Youden Thr: %.3f\n", auc_log_bal, thr_bal))
print(log_cm_bal$byClass[c("Sensitivity","Specificity","Balanced Accuracy")])

# ==========================================================
# LDA on balanced test  (make columns match training exactly)
# ==========================================================

# 1) Dummy the balanced test with the SAME recipe used for training
X_test_bal <- as.data.frame(predict(dv, newdata = test_bal))

# 2) Force the columns to match the LDA training matrix exactly
keep_cols <- colnames(X_train2)

# Add any missing columns (fill with 0), then reorder to match training
missing_cols <- setdiff(keep_cols, colnames(X_test_bal))
for (m in missing_cols) X_test_bal[[m]] <- 0
X_test_bal <- X_test_bal[, keep_cols, drop = FALSE]

# 3) (Optional) Drop any extra columns that appear in test but not in training
extra_cols <- setdiff(colnames(X_test_bal), keep_cols)
if (length(extra_cols) > 0) X_test_bal <- X_test_bal[, keep_cols, drop = FALSE]

# 4) Remove rows with NA predictors and align y
row_ok_test_bal <- stats::complete.cases(X_test_bal) & !is.na(test_bal$y)
X_test_bal2 <- X_test_bal[row_ok_test_bal, , drop = FALSE]
y_test_bal2 <- droplevels(test_bal$y[row_ok_test_bal])

# 5) Predict with LDA and evaluate
lda_probs_bal <- predict(lda_model, newdata = X_test_bal2)$posterior[, "yes"]

roc_lda_bal <- pROC::roc(y_test_bal2, lda_probs_bal, levels = c("no","yes"), quiet = TRUE)
auc_lda_bal <- as.numeric(pROC::auc(roc_lda_bal))
thr_lda_bal <- as.numeric(pROC::coords(roc_lda_bal, "best", best.method = "youden", ret = "threshold"))
if (length(thr_lda_bal) > 1) thr_lda_bal <- mean(thr_lda_bal)

lda_pred_bal <- factor(ifelse(lda_probs_bal >= thr_lda_bal, "yes", "no"), levels = c("no","yes"))
lda_cm_bal   <- caret::confusionMatrix(lda_pred_bal, y_test_bal2, positive = "yes")

cat("\n=== LDA (Balanced Test) ===\n")
cat(sprintf("AUC: %.4f | Youden Thr: %.3f\n", auc_lda_bal, thr_lda_bal))
print(lda_cm_bal$byClass[c("Sensitivity","Specificity","Balanced Accuracy")])


# ---- Part C: Insights ----
cat("\n=== Part C: Insights ===\n")
log_tidy <- broom::tidy(log_model) %>%
  dplyr::mutate(OR = exp(estimate), abs_z = abs(statistic)) %>%
  dplyr::arrange(desc(abs_z))

cat("\nTop 10 effects by |z| (logistic):\n")
print(log_tidy %>% dplyr::slice(1:10) %>% dplyr::select(term, estimate, OR, statistic, p.value))

top_pos <- log_tidy %>% dplyr::filter(estimate > 0) %>%
  dplyr::slice_max(order_by = estimate, n = 5) %>% dplyr::pull(term)
cat("\nPersonas likely to respond (heuristic from positive terms):\n"); print(top_pos)
cat("\nNotes:\n- Positive estimate (OR>1) increases odds of subscription; negative decreases.\n- Factor terms interpret relative to baseline level.\n")

# ---- Part D: Strategy & Recommendations ----
cat("\n=== Part D: Strategy & Recommendations ===\n")
lift_df <- tibble(prob = log_probs_ok, actual = as.integer(y_ok == "yes")) %>%
  dplyr::arrange(dplyr::desc(prob)) %>%
  dplyr::mutate(rank = row_number(),
                frac = rank / n(),
                decile = ntile(-prob, 10))  # 1 = highest scores

overall_rate <- mean(lift_df$actual)
cat(sprintf("Overall conversion rate in test: %.4f\n", overall_rate))

top20 <- lift_df %>% dplyr::filter(frac <= 0.20)
top20_rate <- mean(top20$actual)
lift_top20 <- top20_rate / overall_rate
cat(sprintf("If you contact top 20%%: expected conversion rate = %.4f | Lift = %.2fx over random.\n",
            top20_rate, lift_top20))

decile_lift <- lift_df %>%
  dplyr::group_by(decile) %>%
  dplyr::summarise(n = dplyr::n(),
                   resp_rate = mean(actual),
                   cum_n = cumsum(n),
                   cum_resp = cumsum(n*resp_rate),
                   .groups = "drop") %>%
  dplyr::mutate(cum_frac = cum_n / sum(n),
                lift = resp_rate / overall_rate,
                cum_resp_rate = cum_resp / cum_n)

cat("\nDecile lift table (top=1 is highest score):\n"); print(decile_lift)

cat("\nEthical considerations:\n")
cat("- Audit for disparate impact across age, job, education.\n")
cat("- Avoid excluding protected groups based on historical bias.\n")
cat("- Consider fairness constraints or post-model reviews before deployment.\n")

cat("\n=== End ===\n")





# =========================
# Part E: Plots / Visuals
# =========================
library(ggplot2)

# --- 1) ROC curves (Logistic vs LDA) ---
roc_df_log <- data.frame(
  fpr = 1 - roc_log$specificities,
  tpr = roc_log$sensitivities,
  model = "Logistic"
)
roc_df_lda <- data.frame(
  fpr = 1 - roc_lda$specificities,
  tpr = roc_lda$sensitivities,
  model = "LDA"
)
roc_df_both <- rbind(roc_df_log, roc_df_lda)

ggplot(roc_df_both, aes(x = fpr, y = tpr, color = model)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  labs(title = sprintf("ROC Curves (AUC: Logistic=%.3f, LDA=%.3f)", auc_log, auc_lda),
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()

# --- 2) Sensitivity & Specificity vs Threshold (Logistic) ---
coords_log <- as.data.frame(pROC::coords(
  roc_log, x = "all",
  ret = c("threshold","sensitivity","specificity"),
  transpose = FALSE
))
ggplot(coords_log, aes(x = threshold)) +
  geom_line(aes(y = sensitivity, linetype = "Sensitivity"), linewidth = 1) +
  geom_line(aes(y = specificity, linetype = "Specificity"), linewidth = 1) +
  geom_vline(xintercept = thr_you, linetype = "dashed") +
  annotate("text", x = thr_you, y = 0.03, label = "Youden thr", angle = 90, vjust = -0.3, size = 3) +
  { if (exists("thr_target")) geom_vline(xintercept = thr_target, linetype = "dashed", alpha = 0.7) } +
  labs(title = "Sensitivity & Specificity vs Threshold (Logistic)",
       x = "Threshold", y = "Rate") +
  theme_minimal() +
  scale_linetype_manual(values = c("Sensitivity" = "solid", "Specificity" = "dotdash"))

# --- 3) Accuracy vs Threshold (Logistic) ---
coords_log$accuracy <- sapply(coords_log$threshold, function(t) {
  pred <- factor(ifelse(log_probs_ok >= t, "yes", "no"), levels = c("no","yes"))
  mean(pred == y_ok)
})
ggplot(coords_log, aes(x = threshold, y = accuracy)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = thr_you, linetype = "dashed") +
  labs(title = "Accuracy vs Threshold (Logistic)",
       x = "Threshold", y = "Accuracy") +
  theme_minimal()

# --- 4) Confusion matrix heatmap at Youden threshold (Logistic) ---
cm_mat <- as.matrix(log_cm_you$table)  # rows = Prediction, cols = Reference
cm_df  <- as.data.frame(as.table(cm_mat))
colnames(cm_df) <- c("Prediction", "Reference", "Freq")
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", fontface = "bold") +
  scale_fill_gradient(low = "#6baed6", high = "#08519c") +
  labs(title = "Confusion Matrix (Logistic @ Youden threshold)") +
  theme_minimal()

# --- 5) Decile lift chart (response rate by decile) ---
ggplot(decile_lift, aes(x = factor(decile), y = resp_rate)) +
  geom_col() +
  geom_hline(yintercept = overall_rate, linetype = "dashed") +
  labs(title = "Response Rate by Score Decile (1 = highest scores)",
       x = "Decile", y = "Response Rate") +
  theme_minimal()

# --- 6) Cumulative gains curve (Logistic) ---
# Build cumulative gains from your lift_df (already sorted by prob desc)
gains_df <- lift_df %>%
  arrange(desc(prob)) %>%
  mutate(cum_positives = cumsum(actual),
         total_positives = sum(actual),
         cum_frac_pop = row_number()/n(),
         cum_frac_pos = cum_positives/total_positives)

ggplot(gains_df, aes(x = cum_frac_pop, y = cum_frac_pos)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  labs(title = "Cumulative Gains Curve (Logistic)",
       x = "Fraction of Population Contacted",
       y = "Fraction of Positive Responses Captured") +
  theme_minimal()

