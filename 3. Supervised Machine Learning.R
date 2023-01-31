#Installing required libraies
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(ModelMetrics)) install.packages("ModelMetrics", repos = "http://cran.us.r-project.org")

#loading libraries
library(dplyr)
library(caret)
library(e1071)
library(randomForest)
library(gbm)
library(glmnet)
library(ModelMetrics)
library(reshape2)

#reading the dataset that 
df <- read.csv("cleaned_data.csv", header = T, sep = ',')

#Separating Y and X variables.
Y <- log(df[, 'price'])
X <- df[, colnames(df) != 'price']

#Creating Train and Test sets.
splitIndex <- createDataPartition(Y, p = 0.8, list = FALSE)
X_train <- X[splitIndex, ]
Y_train <- Y[splitIndex]
X_test <- X[-splitIndex, ]
Y_test <- Y[-splitIndex]

# Linear Regression
lin_reg <- train(X_train, Y_train, method = "lm")
lin_reg_pred_train <- predict(lin_reg, X_train)
lin_reg_pred_test <- predict(lin_reg, X_test)

# Lasso Regression
lasso_reg <- train(X_train, Y_train, method = "glmnet", tuneGrid = expand.grid(alpha = 0.01, lambda = 0.1))
lasso_reg_pred_train <- predict(lasso_reg, X_train)
lasso_reg_pred_test <- predict(lasso_reg, X_test)

# Decision Trees
dt <- train(X_train, Y_train, method = "rpart")
dt_pred_train <- predict(dt, X_train)
dt_pred_test <- predict(dt, X_test)

# Random Forest
rf <- randomForest(x = X_train, y = Y_train)
rf_pred_train <- predict(rf, X_train)
rf_pred_test <- predict(rf, X_test)

# Gradient Boosting
gb <- gbm(Y_train ~ ., data = X_train, distribution = "gaussian", n.trees = 500)
gb_pred_train <- predict(gb, X_train, n.trees = 500)
gb_pred_test <- predict(gb, X_test, n.trees = 500)


#Compiling results from regressions in a table.

#to calculate R-square
rsq <- function (x, y) cor(x, y) ^ 2

Evaluation_Table <- data.frame(Algorithm = c("Linear Regression", "Lasso Regression", 
                                             "Decision Trees", "Random Forest", "Gradient Boosting"),
                               R_square_train = c(rsq(lin_reg_pred_train, Y_train), rsq(lasso_reg_pred_train, Y_train), rsq(dt_pred_train, Y_train), rsq(rf_pred_train, Y_train), rsq(gb_pred_train, Y_train)),
                               
                               R_square_test = c(rsq(lin_reg_pred_test, Y_test), rsq(lasso_reg_pred_test, Y_test), rsq(dt_pred_test, Y_test), rsq(rf_pred_test, Y_test), rsq(gb_pred_test, Y_test)),
                               
                               MAE = c(MAE(lin_reg_pred_train, Y_train), MAE(lasso_reg_pred_train, Y_train), MAE(dt_pred_train, Y_train), MAE(rf_pred_train, Y_train), MAE(gb_pred_train, Y_train)),
                               
                               MSE = c(mse(lin_reg_pred_train, Y_train), mse(lasso_reg_pred_train, Y_train), mse(dt_pred_train, Y_train), mse(rf_pred_train, Y_train), mse(gb_pred_train, Y_train)),
                               
                               RMSE = c(RMSE(lin_reg_pred_train, Y_train), RMSE(lasso_reg_pred_train, Y_train), RMSE(dt_pred_train, Y_train), RMSE(rf_pred_train, Y_train), RMSE(gb_pred_train, Y_train))
)
#Results of each Regression
Evaluation_Table