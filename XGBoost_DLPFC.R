library(xgboost)
library(caret) 
library(doParallel)
library(foreach)

# of cores from SLURM script
args <- commandArgs(trailingOnly = TRUE)
numCores <- as.numeric(args[1])

# Parallel processing
cl <- makeCluster(numCores)
registerDoParallel(cl)

# Load data, filter and format
load("GTEx_regressed_lmm_RSEM_final_CNS.RData")

datMeta <- datMeta[datMeta$region == "Brain_Frontal_Cortex_BA9",]

datExpr.reg <- datExpr.norm[,colnames(datExpr.norm) %in% datMeta$SAMPID]

datExpr.reg.df <- as.data.frame(t(datExpr.reg))

# List to store results
results <- list()

# Parallelize the loop
results <- foreach(i = 1:ncol(datExpr.reg.df), .combine = rbind, .packages = c('xgboost', 'caret')) %dopar% {
  
  # Set the target 
  target <- datExpr.reg.df[, i]
  
  # Extract and scale the features 
  features <- datExpr.reg.df[, -i]
  
  features_scaled <- scale(features)
  
  # Combine scaled features and target into a dataframe before splitting
  data_scaled <- data.frame(target = target, features_scaled)
  
  # Set seed for reproducibility
  set.seed(123)
  
  # Split the data into training (80%) and test (20%)
  trainIndex <- createDataPartition(data_scaled$target, p = .8, list = FALSE, times = 1)
  trainData <- data_scaled[trainIndex, ]
  testData <- data_scaled[-trainIndex, ]
  
  # Separate the target and features for the training and test sets
  train_features <- as.matrix(trainData[, -1])
  train_target <- trainData$target
  
  test_features <- as.matrix(testData[, -1])
  test_target <- testData$target
  
  # Create DMatrix for training and test data
  dtrain <- xgb.DMatrix(data = train_features, label = train_target)
  dtest <- xgb.DMatrix(data = test_features, label = test_target)

  # Set model parameters (previously identified using a grid search)
  params <- list(
    objective = "reg:squarederror", 
    max_depth = 6, 
    eta = 0.1, 
    nthread = 1,  # already parallelizing in the SLURM script
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 7,
    gamma = 0,
    eval_metric = "rmse"
  )
  
  # Train model with early stopping
  xgb_model <- xgb.train(
    params = params, 
    data = dtrain, 
    nrounds = 100, 
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = 10,
    verbose = 0
  )
  
  # Predict on test data
  predictions <- predict(xgb_model, test_features)
  
  # Calculate RMSE
  rmse <- sqrt(mean((predictions - test_target)^2))
  
  # Calculate the mean of the actual values
  mean_target <- mean(test_target)
  
  # Calculate SS_res (Residual Sum of Squares)
  ss_res <- sum((test_target - predictions)^2)
  
  # Calculate SS_tot (Total Sum of Squares)
  ss_tot <- sum((test_target - mean_target)^2)
  
  # Calculate R-squared
  r_squared <- 1 - (ss_res / ss_tot)
  
  # Calculate NRMSE (Normalized RMSE)
  nrmse <- rmse / (max(test_target) - min(test_target))
  
  # Calculate CV(RMSE) (Coefficient of Variation of RMSE)
  cv_rmse <- rmse / mean_target
  
  # Calculate the standard deviation of the actual values
  sd_target <- sd(test_target)
  
  # Calculate CV of the outcome (coefficient of variation)
  cv_outcome <- sd_target / mean_target
  
  # Store the results in the list
  results[[i]] <- data.frame(
    Outcome = colnames(datExpr.reg.df)[i],
    RMSE = rmse,
    NRMSE = nrmse,
    CV_RMSE = cv_rmse,
    R_squared = r_squared,
    CV_Outcome = cv_outcome  
  )
}

# Stop the cluster
stopCluster(cl)


write.csv(results, "XGBoost_GTEx_DLPFC_Aug12th2024.csv", row.names = FALSE)
