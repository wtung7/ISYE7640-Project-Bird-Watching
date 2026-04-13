# ============================================================
# ISyE 7406 - Species Distribution Modeling
# Full Modeling Script: Tuning + SHAP + Plots
# ============================================================
# Research Questions:
#   RQ1: How accurately can each model predict species presence?
#   RQ2: Does accuracy vary across species with different habitat specializations?
#   RQ3: Which environmental variables are the most important predictors?
#   RQ4: Does model complexity improve performance, and at what cost?
#
# Models: Logistic Regression, LASSO, Random Forest, XGBoost
# Tuning: Random search cross-validation
# Importance: Built-in (RF mean decrease accuracy, XGBoost Gain) + SHAP
# Output: CSVs + plots for report
# ============================================================

# ---- Install & Load Packages -------------------------------
install.packages(c("glmnet", "randomForest", "xgboost", "pROC",
                   "ggplot2", "dplyr", "tidyr", "caret", "shapviz",
                   "gridExtra", "ggcorrplot", "forcats", "tibble"))

library(glmnet)
library(randomForest)
library(xgboost)
library(pROC)
library(ggplot2)
library(dplyr)
library(tidyr)
library(caret)
library(shapviz)
library(gridExtra)
library(forcats)
library(tibble)

# ---- Paths -------------------------------------------------
base_path   <- "/Users/will2/CS Side Progects/ISYE7640-Project-Bird-Watching" # <-- UPDATE THIS TO YOUR BASE PATH
data_path   <- file.path(base_path, "updated_data_for_readability")
output_path <- file.path(base_path, "model_outputs")
plot_path   <- file.path(output_path, "plots")
dir.create(output_path, showWarnings = FALSE)
dir.create(plot_path,   showWarnings = FALSE)

# ---- Species list ------------------------------------------
species_files <- list(
  "American Robin"     = "american_robin_clean.csv",
  "Steller's Jay"      = "stellers_jay_clean.csv",
  "Red-tailed Hawk"    = "redtailed_hawk_clean.csv",
  "Spotted Towhee"     = "spotted_towhee_clean.csv",
  "Western Meadowlark" = "western_meadowlark_clean.csv"
)

# ---- Settings ----------------------------------------------
K        <- 5    # CV folds
N_SEARCH <- 20   # random search iterations per model
set.seed(7406)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

compute_metrics <- function(actual, predicted_prob, threshold = 0.5) {
  predicted_class <- as.integer(predicted_prob >= threshold)
  acc <- mean(predicted_class == actual)
  auc <- as.numeric(auc(roc(actual, predicted_prob, quiet = TRUE)))
  list(accuracy = acc, auc = auc)
}

# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================

load_species_data <- function(csv_file) {
  dat <- read.csv(file.path(data_path, csv_file))

  # drop non-predictor columns
  drop_cols <- c("species", "observation_date", "protocol_name",
                 "duration_minutes", "effort_distance_km",
                 "number_observers", "all_species_reported",
                 "longitude", "latitude")
  drop_cols <- drop_cols[drop_cols %in% names(dat)]
  dat <- dat[, !(names(dat) %in% drop_cols)]

  # convert land_cover to numeric for tree models
  dat$land_cover <- as.integer(as.character(dat$land_cover))
  dat <- na.omit(dat)
  dat
}

# ============================================================
# STEP 2: RANDOM SEARCH TUNING FUNCTIONS
# ============================================================

# ---- LASSO tuning ------------------------------------------
tune_lasso <- function(X_train, y_train) {
  cat("    Tuning LASSO...\n")
  alpha_grid  <- runif(N_SEARCH, 0.5, 1.0)
  best_auc    <- -Inf
  best_alpha  <- 1
  best_lambda <- NULL

  # store full CV curve for the best alpha (for tuning plot)
  best_cv_fit <- NULL

  for (a in alpha_grid) {
    cv_fit <- cv.glmnet(X_train, y_train, alpha = a,
                        family = "binomial", nfolds = 3,
                        type.measure = "auc")
    if (max(cv_fit$cvm) > best_auc) {
      best_auc    <- max(cv_fit$cvm)
      best_alpha  <- a
      best_lambda <- cv_fit$lambda.min
      best_cv_fit <- cv_fit
    }
  }
  cat("    Best alpha:", round(best_alpha, 3),
      "| Best lambda:", round(best_lambda, 5), "\n")

  # build lambda curve dataframe for plotting
  lambda_df <- data.frame(
    log_lambda = log(best_cv_fit$lambda),
    auc        = best_cv_fit$cvm,
    auc_hi     = best_cv_fit$cvm + best_cv_fit$cvsd,
    auc_lo     = best_cv_fit$cvm - best_cv_fit$cvsd
  )

  list(alpha = best_alpha, lambda = best_lambda, lambda_df = lambda_df)
}

# ---- Random Forest tuning ----------------------------------
tune_rf <- function(X_train, y_train) {
  cat("    Tuning Random Forest...\n")

  mtry_vals     <- sample(2:ncol(X_train), N_SEARCH, replace = TRUE)
  ntree_vals    <- sample(c(100, 200, 300, 500), N_SEARCH, replace = TRUE)
  nodesize_vals <- sample(c(1, 3, 5, 10), N_SEARCH, replace = TRUE)

  best_auc    <- -Inf
  best_params <- list(mtry = floor(sqrt(ncol(X_train))),
                      ntree = 500, nodesize = 1)

  # record all search results for tuning plot
  search_log <- data.frame(
    iter     = 1:N_SEARCH,
    mtry     = mtry_vals,
    ntree    = ntree_vals,
    nodesize = nodesize_vals,
    auc      = NA_real_
  )

  for (i in 1:N_SEARCH) {
    rf_fit <- randomForest(x = X_train, y = as.factor(y_train),
                            mtry     = mtry_vals[i],
                            ntree    = ntree_vals[i],
                            nodesize = nodesize_vals[i])
    pred    <- predict(rf_fit, X_train, type = "prob")[, 2]
    auc_val <- as.numeric(auc(roc(y_train, pred, quiet = TRUE)))
    search_log$auc[i] <- auc_val

    if (auc_val > best_auc) {
      best_auc    <- auc_val
      best_params <- list(mtry     = mtry_vals[i],
                          ntree    = ntree_vals[i],
                          nodesize = nodesize_vals[i])
    }
  }
  cat("    Best mtry:", best_params$mtry,
      "| ntree:", best_params$ntree,
      "| nodesize:", best_params$nodesize, "\n")

  c(best_params, list(search_log = search_log))
}

# ---- XGBoost tuning ----------------------------------------
tune_xgb <- function(X_train, y_train) {
  cat("    Tuning XGBoost...\n")
  dtrain <- xgb.DMatrix(data = X_train, label = y_train)

  best_auc <- -Inf
  best_params <- list(
    eta = 0.1, max_depth = 6, subsample = 0.8,
    colsample_bytree = 0.8, min_child_weight = 1,
    objective = "binary:logistic", eval_metric = "auc",
    nrounds = 100
  )

  # record all search results for tuning plot
  search_log <- data.frame(
    iter      = 1:N_SEARCH,
    eta       = NA_real_,
    max_depth = NA_integer_,
    auc       = NA_real_
  )

  for (i in 1:N_SEARCH) {
    params <- list(
      objective        = "binary:logistic",
      eval_metric      = "auc",
      eta              = runif(1, 0.01, 0.3),
      max_depth        = sample(3:8, 1),
      subsample        = runif(1, 0.6, 1.0),
      colsample_bytree = runif(1, 0.6, 1.0),
      min_child_weight = sample(1:10, 1)
    )
    nrounds <- sample(c(50, 100, 150, 200), 1)

    cv_result <- tryCatch(
      xgb.cv(params = params, data = dtrain, nrounds = nrounds,
              nfold = 3, verbose = FALSE, early_stopping_rounds = 10),
      error = function(e) NULL
    )

    if (is.null(cv_result)) next

    auc_col <- cv_result$evaluation_log$test_auc_mean
    if (is.null(auc_col) || length(auc_col) == 0) next

    auc_val   <- max(auc_col, na.rm = TRUE)
    best_iter <- cv_result$best_iteration
    if (is.null(best_iter) || length(best_iter) == 0 || is.na(best_iter)) {
      best_iter <- which.max(auc_col)
    }
    if (length(best_iter) == 0 || is.na(best_iter) || best_iter <= 0) {
      best_iter <- nrounds
    }

    search_log$eta[i]       <- params$eta
    search_log$max_depth[i] <- params$max_depth
    search_log$auc[i]       <- auc_val

    if (auc_val > best_auc) {
      best_auc    <- auc_val
      best_params <- c(params, list(nrounds = best_iter))
    }
  }

  cat("    Best eta:", round(best_params$eta, 3),
      "| max_depth:", best_params$max_depth,
      "| nrounds:", best_params$nrounds, "\n")

  c(best_params, list(search_log = search_log))
}

# ============================================================
# STEP 3: MAIN MODELING FUNCTION
# ============================================================

run_models <- function(species_name, csv_file) {

  cat("\n============================================================\n")
  cat("Modeling:", species_name, "\n")
  cat("============================================================\n")

  dat <- load_species_data(csv_file)
  y   <- dat$presence
  X   <- as.matrix(dat[, names(dat) != "presence"])

  cat("  Records:", nrow(dat),
      "| Presences:", sum(y == 1),
      "| Absences:", sum(y == 0),
      "| Predictors:", ncol(X), "\n")

  folds <- createFolds(y, k = K, list = TRUE, returnTrain = FALSE)

  # storage for per-fold metrics
  metrics <- data.frame(
    fold     = rep(1:K, 4),
    model    = rep(c("Logistic Regression", "LASSO",
                     "Random Forest", "XGBoost"), each = K),
    accuracy = NA,
    auc      = NA
  )

  best_hp <- list()

  # ---- Tune on 80% sample before CV -------------------------
  cat("\n  Running hyperparameter tuning...\n")
  train_sample <- sample(1:nrow(dat), floor(0.8 * nrow(dat)))
  X_tune <- X[train_sample, ]
  y_tune <- y[train_sample]

  lasso_hp <- tune_lasso(X_tune, y_tune)
  rf_hp    <- tune_rf(X_tune, y_tune)
  xgb_hp   <- tune_xgb(X_tune, y_tune)

  best_hp[["LASSO"]]          <- lasso_hp
  best_hp[["Random Forest"]]  <- rf_hp
  best_hp[["XGBoost"]]        <- xgb_hp

  # tag tuning logs with species for combined plots
  lasso_hp$lambda_df$species <- species_name
  rf_hp$search_log$species   <- species_name
  xgb_hp$search_log$species  <- species_name

  # ---- 5-fold CV with tuned hyperparameters ------------------
  cat("\n  Running 5-fold CV with tuned parameters...\n")

  for (k in 1:K) {
    cat("  Fold", k, "of", K, "\n")

    test_idx  <- folds[[k]]
    train_idx <- setdiff(1:nrow(dat), test_idx)

    X_train <- X[train_idx, ]
    X_test  <- X[test_idx,  ]
    y_train <- y[train_idx]
    y_test  <- y[test_idx]

    df_train <- data.frame(presence = y_train, X_train)
    df_test  <- data.frame(presence = y_test,  X_test)

    # -- 1. Logistic Regression --------------------------------
    lr_mod  <- glm(presence ~ ., data = df_train, family = binomial)
    lr_pred <- predict(lr_mod, newdata = df_test, type = "response")
    m <- compute_metrics(y_test, lr_pred)
    metrics[metrics$fold == k & metrics$model == "Logistic Regression",
            c("accuracy", "auc")] <- c(m$accuracy, m$auc)

    # -- 2. LASSO ----------------------------------------------
    lasso_mod  <- glmnet(X_train, y_train, alpha = lasso_hp$alpha,
                          family = "binomial", lambda = lasso_hp$lambda)
    lasso_pred <- predict(lasso_mod, newx = X_test, type = "response")[, 1]
    m <- compute_metrics(y_test, lasso_pred)
    metrics[metrics$fold == k & metrics$model == "LASSO",
            c("accuracy", "auc")] <- c(m$accuracy, m$auc)

    # -- 3. Random Forest --------------------------------------
    rf_mod  <- randomForest(x = X_train, y = as.factor(y_train),
                             mtry      = rf_hp$mtry,
                             ntree     = rf_hp$ntree,
                             nodesize  = rf_hp$nodesize,
                             importance = TRUE)
    rf_pred <- predict(rf_mod, newdata = X_test, type = "prob")[, 2]
    m <- compute_metrics(y_test, rf_pred)
    metrics[metrics$fold == k & metrics$model == "Random Forest",
            c("accuracy", "auc")] <- c(m$accuracy, m$auc)

    # -- 4. XGBoost --------------------------------------------
    dtrain  <- xgb.DMatrix(data = X_train, label = y_train)
    dtest   <- xgb.DMatrix(data = X_test,  label = y_test)
    xgb_params <- xgb_hp[!names(xgb_hp) %in% c("nrounds", "search_log")]
    xgb_mod <- xgb.train(params  = xgb_params,
                          data    = dtrain,
                          nrounds = xgb_hp$nrounds,
                          verbose = 0)
    xgb_pred <- predict(xgb_mod, dtest)
    m <- compute_metrics(y_test, xgb_pred)
    metrics[metrics$fold == k & metrics$model == "XGBoost",
            c("accuracy", "auc")] <- c(m$accuracy, m$auc)
  }

  # ---- Summarize CV results ----------------------------------
  cv_summary <- metrics |>
    group_by(model) |>
    summarise(
      mean_auc      = round(mean(auc),      4),
      sd_auc        = round(sd(auc),        4),
      mean_accuracy = round(mean(accuracy), 4),
      sd_accuracy   = round(sd(accuracy),   4),
      .groups = "drop"
    ) |>
    mutate(species = species_name)

  cat("\n  CV Results:\n")
  print(cv_summary[, c("model", "mean_auc", "sd_auc",
                        "mean_accuracy", "sd_accuracy")])

  # ============================================================
  # STEP 4: FEATURE IMPORTANCE ON FULL DATA
  # ============================================================
  cat("\n  Computing feature importance...\n")

  # -- RF importance --
  rf_full <- randomForest(x = X, y = as.factor(y),
                           mtry      = rf_hp$mtry,
                           ntree     = rf_hp$ntree,
                           nodesize  = rf_hp$nodesize,
                           importance = TRUE)

  rf_imp_df <- importance(rf_full, type = 1) |>
    as.data.frame() |>
    rownames_to_column("variable") |>
    rename(importance = MeanDecreaseAccuracy) |>
    mutate(model = "Random Forest", species = species_name)

  # -- XGBoost importance --
  dtrain_full <- xgb.DMatrix(data = X, label = y)
  xgb_params  <- xgb_hp[!names(xgb_hp) %in% c("nrounds", "search_log")]
  xgb_full    <- xgb.train(params  = xgb_params,
                             data    = dtrain_full,
                             nrounds = xgb_hp$nrounds,
                             verbose = 0)

  xgb_imp    <- xgb.importance(model = xgb_full)
  xgb_imp_df <- data.frame(
    variable   = xgb_imp$Feature,
    importance = xgb_imp$Gain,
    model      = "XGBoost",
    species    = species_name
  )

  importance_df <- bind_rows(rf_imp_df, xgb_imp_df)

  # ============================================================
  # STEP 5: SHAP VALUES
  # ============================================================
  cat("  Computing SHAP values...\n")

  shap_idx <- sample(1:nrow(X), min(500, nrow(X)))
  X_shap   <- X[shap_idx, ]

  shap_xgb <- shapviz(xgb_full, X_pred = X_shap)

  shap_plot <- sv_importance(shap_xgb, kind = "beeswarm", max_display = 15) +
    labs(title    = paste("SHAP Values -", species_name),
         subtitle = "XGBoost model") +
    theme_bw()

  ggsave(file.path(plot_path,
                   paste0(gsub("[^a-z]", "_", tolower(species_name)),
                          "_shap.png")),
         shap_plot, width = 8, height = 6, dpi = 150)

  shap_means <- colMeans(abs(shap_xgb$S)) |>
    sort(decreasing = TRUE) |>
    as.data.frame() |>
    rownames_to_column("variable")
  names(shap_means)[2] <- "mean_abs_shap"
  shap_means$species   <- species_name

  list(
    cv_summary  = cv_summary,
    importance  = importance_df,
    shap        = shap_means,
    best_hp     = best_hp,
    rf_model    = rf_full,
    xgb_model   = xgb_full,
    lasso_curve = lasso_hp$lambda_df,
    rf_search   = rf_hp$search_log,
    xgb_search  = xgb_hp$search_log
  )
}

# ============================================================
# STEP 6: RUN ALL SPECIES
# ============================================================
all_results <- list()

for (sp in names(species_files)) {
  all_results[[sp]] <- run_models(sp, species_files[[sp]])
}

# combine results
cv_all       <- bind_rows(lapply(all_results, `[[`, "cv_summary"))
imp_all      <- bind_rows(lapply(all_results, `[[`, "importance"))
shap_all     <- bind_rows(lapply(all_results, `[[`, "shap"))
lasso_curves <- bind_rows(lapply(all_results, `[[`, "lasso_curve"))
rf_searches  <- bind_rows(lapply(all_results, `[[`, "rf_search"))
xgb_searches <- bind_rows(lapply(all_results, `[[`, "xgb_search"))

hp_rows <- lapply(names(all_results), function(sp) {
  hp <- all_results[[sp]]$best_hp
  data.frame(
    species      = sp,
    lasso_alpha  = round(hp$LASSO$alpha,  3),
    lasso_lambda = round(hp$LASSO$lambda, 5),
    rf_mtry      = hp[["Random Forest"]]$mtry,
    rf_ntree     = hp[["Random Forest"]]$ntree,
    rf_nodesize  = hp[["Random Forest"]]$nodesize,
    xgb_eta      = round(hp$XGBoost$eta, 3),
    xgb_depth    = hp$XGBoost$max_depth,
    xgb_nrounds  = hp$XGBoost$nrounds
  )
})
hp_all <- bind_rows(hp_rows)

# ============================================================
# STEP 7: SAVE CSVs
# ============================================================
write.csv(cv_all,   file.path(output_path, "model_performance.csv"),    row.names = FALSE)
write.csv(imp_all,  file.path(output_path, "feature_importance.csv"),   row.names = FALSE)
write.csv(shap_all, file.path(output_path, "shap_importance.csv"),      row.names = FALSE)
write.csv(hp_all,   file.path(output_path, "best_hyperparameters.csv"), row.names = FALSE)

cat("\nCSVs saved to:", output_path, "\n")

# ============================================================
# STEP 8: PLOTS FOR REPORT
# ============================================================
cat("\nGenerating report plots...\n")

# ---- RQ1 & RQ2: AUC by model and species -------------------
p_auc <- ggplot(cv_all, aes(x = fct_reorder(model, mean_auc),
                              y = mean_auc, fill = model)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin = mean_auc - sd_auc,
                    ymax = mean_auc + sd_auc), width = 0.3) +
  facet_wrap(~ species, ncol = 2) +
  labs(title    = "RQ1 & RQ2: Mean AUC-ROC by Model and Species",
       subtitle = paste0(K, "-fold CV with random search tuning"),
       x = NULL, y = "Mean AUC-ROC") +
  theme_bw() +
  theme(axis.text.x  = element_text(angle = 30, hjust = 1),
        legend.position = "none") +
  ylim(0.5, 1.0)

ggsave(file.path(plot_path, "rq1_rq2_auc_by_model_species.png"),
       p_auc, width = 10, height = 8, dpi = 150)

# ---- RQ2: AUC heatmap --------------------------------------
p_heatmap <- ggplot(cv_all, aes(x = model, y = species, fill = mean_auc)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(mean_auc, 3)), size = 3.5) +
  scale_fill_gradient(low = "#FFF3E0", high = "#1B5E20",
                      limits = c(0.5, 1.0), name = "Mean AUC") +
  labs(title = "RQ2: AUC Heatmap -- Habitat Specialist vs Generalist",
       x = "Model", y = "Species") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(file.path(plot_path, "rq2_auc_heatmap.png"),
       p_heatmap, width = 8, height = 5, dpi = 150)

# ---- RQ3: Top 10 RF feature importance per species ---------
rf_imp_top <- imp_all |>
  filter(model == "Random Forest") |>
  group_by(species) |>
  slice_max(importance, n = 10) |>
  ungroup()

p_imp_rf <- ggplot(rf_imp_top,
                    aes(x = reorder(variable, importance),
                        y = importance, fill = species)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ species, scales = "free_y", ncol = 2) +
  coord_flip() +
  labs(title    = "RQ3: Top 10 Predictors by Species (Random Forest)",
       subtitle = "Mean Decrease in Accuracy",
       x = NULL, y = "Importance") +
  theme_bw() +
  theme(legend.position = "none")

ggsave(file.path(plot_path, "rq3_rf_feature_importance.png"),
       p_imp_rf, width = 12, height = 10, dpi = 150)

# ---- RQ3: SHAP importance per species ----------------------
shap_top <- shap_all |>
  group_by(species) |>
  slice_max(mean_abs_shap, n = 10) |>
  ungroup()

p_shap <- ggplot(shap_top,
                  aes(x = reorder(variable, mean_abs_shap),
                      y = mean_abs_shap, fill = species)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ species, scales = "free_y", ncol = 2) +
  coord_flip() +
  labs(title    = "RQ3: Top 10 Predictors by Species (SHAP Values)",
       subtitle = "Mean Absolute SHAP Value -- XGBoost model",
       x = NULL, y = "Mean |SHAP|") +
  theme_bw() +
  theme(legend.position = "none")

ggsave(file.path(plot_path, "rq3_shap_importance.png"),
       p_shap, width = 12, height = 10, dpi = 150)

# ---- RQ4: Complexity vs performance ------------------------
complexity_order <- c("Logistic Regression", "LASSO",
                       "Random Forest", "XGBoost")
cv_all$model <- factor(cv_all$model, levels = complexity_order)

p_complexity <- ggplot(cv_all, aes(x = model, y = mean_auc,
                                    group = species, color = species)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean_auc - sd_auc,
                    ymax = mean_auc + sd_auc), width = 0.2) +
  labs(title    = "RQ4: Model Complexity vs Predictive Performance",
       subtitle = "AUC-ROC across increasing model complexity",
       x = "Model (increasing complexity)",
       y = "Mean AUC-ROC",
       color = "Species") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
  ylim(0.5, 1.0)

ggsave(file.path(plot_path, "rq4_complexity_vs_performance.png"),
       p_complexity, width = 9, height = 6, dpi = 150)

# ---- Accuracy comparison -----------------------------------
p_acc <- ggplot(cv_all, aes(x = fct_reorder(model, mean_accuracy),
                              y = mean_accuracy, fill = model)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy), width = 0.3) +
  facet_wrap(~ species, ncol = 2) +
  labs(title    = "Mean Accuracy by Model and Species",
       subtitle = paste0(K, "-fold CV with random search tuning"),
       x = NULL, y = "Mean Accuracy") +
  theme_bw() +
  theme(axis.text.x  = element_text(angle = 30, hjust = 1),
        legend.position = "none") +
  ylim(0.5, 1.0)

ggsave(file.path(plot_path, "accuracy_by_model_species.png"),
       p_acc, width = 10, height = 8, dpi = 150)

# ---- Tuning Plot 1: LASSO lambda curve (combined) ----------
p_lasso_tune <- ggplot(lasso_curves,
                        aes(x = log_lambda, y = auc)) +
  geom_line(color = "#1565C0", linewidth = 0.8) +
  geom_ribbon(aes(ymin = auc_lo, ymax = auc_hi),
              alpha = 0.15, fill = "#1565C0") +
  geom_vline(data = lasso_curves |>
               group_by(species) |>
               slice_max(auc, n = 1),
             aes(xintercept = log_lambda),
             linetype = "dashed", color = "red", linewidth = 0.7) +
  facet_wrap(~ species, ncol = 2, scales = "free_x") +
  labs(title    = "LASSO Tuning: AUC vs log(lambda)",
       subtitle = "Red dashed line = selected lambda.min",
       x        = "log(lambda)",
       y        = "Cross-validated AUC") +
  theme_bw()

ggsave(file.path(plot_path, "tuning_lasso_lambda_curve.png"),
       p_lasso_tune, width = 10, height = 8, dpi = 150)

# ---- Tuning Plot 2: Random Forest mtry search (combined) ---
p_rf_tune <- ggplot(rf_searches,
                     aes(x = mtry, y = auc, color = species)) +
  geom_point(size = 2.5, alpha = 0.7) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 0.8) +
  facet_wrap(~ species, ncol = 2) +
  labs(title    = "Random Forest Tuning: AUC vs mtry",
       subtitle = "Each point = one random search iteration",
       x        = "mtry (variables sampled per split)",
       y        = "AUC (training)") +
  theme_bw() +
  theme(legend.position = "none")

ggsave(file.path(plot_path, "tuning_rf_mtry_search.png"),
       p_rf_tune, width = 10, height = 8, dpi = 150)

# ---- Tuning Plot 3: XGBoost eta vs AUC (combined) ----------
p_xgb_tune <- ggplot(xgb_searches |> filter(!is.na(auc)),
                      aes(x = eta, y = auc,
                          color = as.factor(max_depth))) +
  geom_point(size = 2.5, alpha = 0.8) +
  facet_wrap(~ species, ncol = 2) +
  scale_color_brewer(palette = "RdYlGn", name = "max_depth") +
  labs(title    = "XGBoost Tuning: AUC vs Learning Rate (eta)",
       subtitle = "Color = max_depth; each point = one random search iteration",
       x        = "eta (learning rate)",
       y        = "Cross-validated AUC") +
  theme_bw()

ggsave(file.path(plot_path, "tuning_xgb_eta_search.png"),
       p_xgb_tune, width = 10, height = 8, dpi = 150)

cat("  Tuning plots saved\n")

# ---- Summary -----------------------------------------------
cat("\nBest Hyperparameters per Species:\n")
print(hp_all)

cat("\n============================================================\n")
cat("All outputs saved to:", output_path, "\n")
cat("\nFiles produced:\n")
cat("  CSVs:\n")
cat("    model_performance.csv    -- mean AUC and accuracy per model per species\n")
cat("    feature_importance.csv   -- RF and XGBoost built-in importance\n")
cat("    shap_importance.csv      -- mean absolute SHAP values per variable\n")
cat("    best_hyperparameters.csv -- tuned parameters per species\n")
cat("\n  Plots (in plots/ subfolder):\n")
cat("    rq1_rq2_auc_by_model_species.png  -- answers RQ1 and RQ2\n")
cat("    rq2_auc_heatmap.png               -- specialist vs generalist heatmap\n")
cat("    rq3_rf_feature_importance.png     -- top predictors (RF)\n")
cat("    rq3_shap_importance.png           -- top predictors (SHAP)\n")
cat("    rq4_complexity_vs_performance.png -- answers RQ4\n")
cat("    accuracy_by_model_species.png     -- secondary metric\n")
cat("    [species]_shap.png                -- per-species SHAP beeswarm plots\n")
cat("    tuning_lasso_lambda_curve.png     -- LASSO lambda CV curve\n")
cat("    tuning_rf_mtry_search.png         -- RF random search: mtry vs AUC\n")
cat("    tuning_xgb_eta_search.png         -- XGBoost random search: eta vs AUC\n")
cat("============================================================\n")