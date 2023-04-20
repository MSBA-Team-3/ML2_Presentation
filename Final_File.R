library(tree)
library(randomForest)
library(xgboost)
library(glmnet)  
library(caret)
library(car)
library (gbm)
library(leaps)
library(corrplot)
rm(list=ls())
setwd("C:/Users/rhlof/Desktop/Spring_23/Machine Learning II/house-prices-advanced-regression-techniques")
data <- read.csv("train.csv")

# Do take the time to understand your data
head(data)

#We don't care about the IDs, will have no impact on home price
ID_labels <- data$Id
data$Id <- NULL

#Which columns have n/a values?
NAcol <- which(colSums(is.na(data)) > 0)
sort(colSums(sapply(data[NAcol], is.na)), decreasing = TRUE)


#For every column with n/a values, we need to set them to either 'None' or 0.
data$PoolQC <- ifelse(is.na(data$PoolQC), 'None', data$PoolQC)
data$MiscFeature <- ifelse(is.na(data$MiscFeature), 'None', data$MiscFeature)
data$Alley <- ifelse(is.na(data$Alley), 'None', data$Alley)
data$Fence <- ifelse(is.na(data$Fence), 'None', data$Fence)
data$FireplaceQu <- ifelse(is.na(data$FireplaceQu), 'None', data$FireplaceQu)
data$GarageType <- ifelse(is.na(data$GarageType), 'None', data$GarageType)
data$GarageFinish <- ifelse(is.na(data$GarageFinish), 'None', data$GarageFinish)
data$GarageQual <- ifelse(is.na(data$GarageQual), 'None', data$GarageQual)
data$GarageCond <- ifelse(is.na(data$GarageCond), 'None', data$GarageCond)
data$BsmtExposure <- ifelse(is.na(data$BsmtExposure), 'None', data$BsmtExposure)
data$BsmtFinType2 <- ifelse(is.na(data$BsmtFinType2), 'None', data$BsmtFinType2)
data$BsmtQual <- ifelse(is.na(data$BsmtQual), 'None', data$BsmtQual)
data$BsmtCond <- ifelse(is.na(data$BsmtCond), 'None', data$BsmtCond)
data$BsmtFinType1 <- ifelse(is.na(data$BsmtFinType1), 'None', data$BsmtFinType1)
data$MasVnrType <- ifelse(is.na(data$MasVnrType), 'None', data$MasVnrType)
data$MasVnrArea <- ifelse(is.na(data$MasVnrArea), 0, data$MasVnrArea)
data$Electrical <- ifelse(is.na(data$Electrical), 'None', data$Electrical)

NAcol <- which(colSums(is.na(data)) > 0)
sort(colSums(sapply(data[NAcol], is.na)), decreasing = TRUE)

#For garage year built, setting it to the year the house was built
data$GarageYrBlt[is.na(data$GarageYrBlt)] <- data$YearBuilt[is.na(data$GarageYrBlt)]

#For Lot Frontage, setting it to the median value for the given neighborhood
for (i in 1:nrow(data)){
  if(is.na(data$LotFrontage[i])){
    data$LotFrontage[i] <- as.integer(median(data$LotFrontage[data$Neighborhood==data$Neighborhood[i]], na.rm=TRUE)) 
  }
}

#Check if we still have remaining n/a values
NAcol <- which(colSums(is.na(data)) > 0)
length(NAcol) #Length = 0, we're good!

#We need to set our columns which are characters to factors
data$MSZoning <- as.factor(data$MSZoning)
data$Street <- as.factor(data$Street)
data$Alley <- as.factor(data$Alley)
data$LotShape <- as.factor(data$LotShape)
data$LandContour <- as.factor(data$LandContour)
data$Utilities <- as.factor(data$Utilities)
data$LotConfig <- as.factor(data$LotConfig)
data$LandSlope <- as.factor(data$LandSlope)
data$Neighborhood <- as.factor(data$Neighborhood)
data$Condition1 <- as.factor(data$Condition1)
data$Condition2 <- as.factor(data$Condition2)
data$BldgType <- as.factor(data$BldgType)
data$HouseStyle <- as.factor(data$HouseStyle)
data$RoofStyle <- as.factor(data$RoofStyle)
data$RoofMatl <- as.factor(data$RoofMatl)
data$Exterior1st <- as.factor(data$Exterior1st)
data$Exterior2nd <- as.factor(data$Exterior2nd)
data$MasVnrType <- as.factor(data$MasVnrType)
data$ExterQual <- as.factor(data$ExterQual)
data$ExterCond <- as.factor(data$ExterCond)
data$Foundation <- as.factor(data$Foundation)
data$BsmtQual <- as.factor(data$BsmtQual)
data$BsmtCond <- as.factor(data$BsmtCond)
data$BsmtExposure <- as.factor(data$BsmtExposure)
data$BsmtFinType1 <- as.factor(data$BsmtFinType1)
data$BsmtFinType2 <- as.factor(data$BsmtFinType2)
data$Heating <- as.factor(data$Heating)
data$HeatingQC <- as.factor(data$HeatingQC)
data$CentralAir <- as.factor(data$CentralAir)
data$Electrical <- as.factor(data$Electrical)
data$KitchenQual <- as.factor(data$KitchenQual)
data$Functional <- as.factor(data$Functional)
data$FireplaceQu <- as.factor(data$FireplaceQu)
data$GarageType <- as.factor(data$GarageType)
data$GarageFinish <- as.factor(data$GarageFinish)
data$GarageQual <- as.factor(data$GarageQual)
data$GarageCond <- as.factor(data$GarageCond)
data$PavedDrive <- as.factor(data$PavedDrive)
data$PoolQC <- as.factor(data$PoolQC)
data$Fence <- as.factor(data$Fence)
data$MiscFeature <- as.factor(data$MiscFeature)
data$SaleType <- as.factor(data$SaleType)
data$SaleCondition <- as.factor(data$SaleCondition)

#Although these may be numeric, they should be factors
data$MSSubClass <- as.factor(data$MSSubClass)
data$MoSold <- as.factor(data$MoSold)

#Feature engineering
#Instead of having four variables which measure bathrooms, we create the TotBathrooms variable
data$TotBathrooms <- data$FullBath + (data$HalfBath*0.5) + data$BsmtFullBath + (data$BsmtHalfBath*0.5)

#Setting variable of whether or not home has been remodeled
data$Remod <- ifelse(data$YearBuilt==data$YearRemodAdd, 'No', 'Yes')
data$Remod = as.factor(data$Remod)

#Getting the age of the home as opposed to the year it was built
data$Age <- as.numeric(data$YrSold-data$YearBuilt)

#Variable signifying whether or not the home is a new build
data$IsNew <- ifelse(data$YrSold==data$YearBuilt, 'Yes', 'No')
data$IsNew = as.factor(data$IsNew)

#Adding SalePrice1 to make sure it is the last variable in our dataframe
data$SalePrice1 <- data$SalePrice

#Dropping unecessary variables given new features
data <- subset(data, select = -c(YearBuilt, YearRemodAdd, SalePrice, FullBath, HalfBath, BsmtFullBath, BsmtHalfBath))


#Find which variables correlate with eachother
numericVars <- which(sapply(data, is.numeric))
factorVars <- which(sapply(data, is.factor))
numVars <- data[, numericVars]
cor_numVar <- cor(numVars, use="pairwise.complete.obs")
cor_numVar
cor_sorted <- as.matrix(sort(cor_numVar[,'SalePrice1'], decreasing = TRUE))
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)

#GarageArea, Garage Cars had high correlations, drop  GarageArea
#TotRmsAbvGrd, GrLivArea, drop TotRmsAbvGrd
#TotalBsmtSF, X1stFlrSF, drop TotalBsmtSF
#GarageYrBlt and Age, drop GarageYrBlt
data <- subset(data, select = -c(GarageArea, TotRmsAbvGrd, TotalBsmtSF, GarageYrBlt))

outliers <- Boxplot(data$SalePrice1 ~ OverallQual, data = data)
outliers
outliers <- as.numeric(outliers)
data <- data[-outliers,]



#Do try out a range of different models
model_rmses <- rep(0,5)

## Model 1: Ridge Model ##

#We know ridge models are useful when dealing with many features,
# so we are trying it out first
# shrinks the coefficient estimates towards zero

#Convert data into a matrix for Ridge model
x <- model.matrix(SalePrice1~., data)[,-1]
y <- data$SalePrice1

#Create our grid which will store lambdas
grid <- 10 ^ seq(10, -2, length=100)

#Do use an appropriate test set
set.seed(1693)
trainIndex <- sample(1:nrow(x), nrow(x)*0.65)

#Create our model, notice alpha = 0
ridge.mod <- glmnet(x[trainIndex, ], 
                    y[trainIndex], 
                    alpha=0, 
                    lambda=grid)
#Cross validate
cv.out <- cv.glmnet(x[trainIndex, ], 
                    y[trainIndex], 
                    alpha=0)
bestlam <- cv.out$lambda.min

#Make our predictions using best lambda value
ridge.pred <- predict(ridge.mod, 
                      s=bestlam,
                      newx=x[-trainIndex, ])

#Find our RMSE
RMSE = sqrt(mean((ridge.pred-y[-trainIndex])^2))
RMSE
#RMSE: 28493.65
model_rmses[1] = RMSE

## Model 2: Lasso Model ##

#Similar to ridge, Lasso models are useful when we have many features
#Also shrinks the coefficient estimates towards zero, but can actually reach zero

set.seed(1693)
#Create lasso model, note alpha = 1
lasso.mod <- glmnet(x[trainIndex, ], 
                    y[trainIndex], 
                    alpha=1,
                    standardize = TRUE,
                    lambda=grid)
#Cross validate
set.seed(1693)
cv.out <- cv.glmnet(x[trainIndex, ], 
                    y[trainIndex], 
                    alpha=1
                    )
bestlam <- cv.out$lambda.min

#Predict using best lambda
set.seed(1693)
lasso.pred <- predict(lasso.mod, 
                      s=bestlam, 
                      newx=x[-trainIndex, ])

#Calculate RMSE
RMSE = sqrt(mean((lasso.pred-y[-trainIndex])^2))
RMSE
#RMSE: 34163.63
model_rmses[2] = RMSE

#So far, our best model is the ridge regression


## Model 3: Tree Models ##

#Try a tree model.  We will be using bagging to implement boostrapping.

#Create our test variable
data_test = data[-trainIndex, "SalePrice1"]

set.seed(1693)
#Create model, notice mtry = 73 so we are bagging
bag <- randomForest(SalePrice1~., data = data, subset = trainIndex, mtry = 73, importance = TRUE)

#make predictions
set.seed(1693)
yhat.bag = predict(bag,newdata=data[-trainIndex,])

#calculate RMSE
RMSE = sqrt(mean((yhat.bag-data_test)^2))
RMSE
#RMSE 24779.01
model_rmses[3] = RMSE

# Bag Model has overtaken the top spot with our lowest RMSE

#Boosted Model

#Create model
set.seed (1693)
boosted <- gbm(SalePrice1~., data = data[trainIndex,])

#Make predictions
set.seed(1693)
preds <- predict(boosted, newdata = data[-trainIndex,])
RMSE = sqrt(mean((preds-data_test)^2))
RMSE
#RMSE: 23901.98
model_rmses[4] = RMSE

## Model 4: XGBoost Model ##

set.seed(1693)
#Set train and test sets
train = data[trainIndex,]
test = data[-trainIndex,]


#Separate predictors and target variable for train set
set.seed(1693)
train_x = data.matrix(train[, -74])
train_y = train[,74]

#Separate predictors and target variable for test set
set.seed(1693)
test_x = data.matrix(test[,-74])
test_y = test[,74]


#Construct dmatrix types for the xgb model
set.seed(1693)
model_train = xgb.DMatrix(data = train_x, label = train_y)
model_test = xgb.DMatrix(data = test_x, label = test_y)

#Create our watchlist, lets us see which nrounds perform best
set.seed(1693)
watchlist = list(train = model_train, test = model_test)

#Create our model with nrounds = 250
set.seed(1693)
model = xgb.train(data = model_train, max.depth = 3, watchlist = watchlist, nrounds = 500)

#Looking through the model, we see that we get the lowest RMSE
level = which.min(model$evaluation_log$test_rmse)
level

#Create model with optimal nround
set.seed(1693)
final = xgboost(data = model_train, max.depth = 3, nrounds = level, verbose = 0)

#Make predictions with model
set.seed(1693)
preds <- predict(final, test_x)


#Calculate RMSE
RMSE = sqrt(mean((preds - test_y)^2))
RMSE
#RMSE is 24301
model_rmses[5] = RMSE

#Which model had the smallest RMSE?
which.min(model_rmses)

#Our fourth model, the boosted model!

#Now to predict values on our test data

#Load in our new data
submission <- read.csv("test.csv")

#Get list of IDs and then remove them
sub_ID_labels <- submission$Id
submission$Id <- NULL

#For every column with n/a values, we need to set them to either 'None' or 0.
submission$PoolQC <- ifelse(is.na(submission$PoolQC), 'None', submission$PoolQC)
submission$MiscFeature <- ifelse(is.na(submission$MiscFeature), 'None', submission$MiscFeature)
submission$Alley <- ifelse(is.na(submission$Alley), 'None', submission$Alley)
submission$Fence <- ifelse(is.na(submission$Fence), 'None', submission$Fence)
submission$FireplaceQu <- ifelse(is.na(submission$FireplaceQu), 'None', submission$FireplaceQu)
submission$GarageType <- ifelse(is.na(submission$GarageType), 'None', submission$GarageType)
submission$GarageFinish <- ifelse(is.na(submission$GarageFinish), 'None', submission$GarageFinish)
submission$GarageQual <- ifelse(is.na(submission$GarageQual), 'None', submission$GarageQual)
submission$GarageCond <- ifelse(is.na(submission$GarageCond), 'None', submission$GarageCond)
submission$BsmtExposure <- ifelse(is.na(submission$BsmtExposure), 'None', submission$BsmtExposure)
submission$BsmtFinType2 <- ifelse(is.na(submission$BsmtFinType2), 'None', submission$BsmtFinType2)
submission$BsmtQual <- ifelse(is.na(submission$BsmtQual), 'None', submission$BsmtQual)
submission$BsmtCond <- ifelse(is.na(submission$BsmtCond), 'None', submission$BsmtCond)
submission$BsmtFinType1 <- ifelse(is.na(submission$BsmtFinType1), 'None', submission$BsmtFinType1)
submission$MasVnrType <- ifelse(is.na(submission$MasVnrType), 'None', submission$MasVnrType)
submission$MasVnrArea <- ifelse(is.na(submission$MasVnrArea), 0, submission$MasVnrArea)
submission$Electrical <- ifelse(is.na(submission$Electrical), 'None', submission$Electrical)

#For garage year built, setting it to the year the house was built
submission$GarageYrBlt[is.na(submission$GarageYrBlt)] <- submission$YearBuilt[is.na(submission$GarageYrBlt)]

#For Lot Frontage, setting it to the median value for the given neighborhood
for (i in 1:nrow(submission)){
  if(is.na(submission$LotFrontage[i])){
    submission$LotFrontage[i] <- as.integer(median(submission$LotFrontage[submission$Neighborhood==submission$Neighborhood[i]], na.rm=TRUE)) 
  }
}

#We need to set our columns which are characters to factors
submission$MSZoning <- as.factor(submission$MSZoning)
submission$Street <- as.factor(submission$Street)
submission$Alley <- as.factor(submission$Alley)
submission$LotShape <- as.factor(submission$LotShape)
submission$LandContour <- as.factor(submission$LandContour)
submission$Utilities <- as.factor(submission$Utilities)
submission$LotConfig <- as.factor(submission$LotConfig)
submission$LandSlope <- as.factor(submission$LandSlope)
submission$Neighborhood <- as.factor(submission$Neighborhood)
submission$Condition1 <- as.factor(submission$Condition1)
submission$Condition2 <- as.factor(submission$Condition2)
submission$BldgType <- as.factor(submission$BldgType)
submission$HouseStyle <- as.factor(submission$HouseStyle)
submission$RoofStyle <- as.factor(submission$RoofStyle)
submission$RoofMatl <- as.factor(submission$RoofMatl)
submission$Exterior1st <- as.factor(submission$Exterior1st)
submission$Exterior2nd <- as.factor(submission$Exterior2nd)
submission$MasVnrType <- as.factor(submission$MasVnrType)
submission$ExterQual <- as.factor(submission$ExterQual)
submission$ExterCond <- as.factor(submission$ExterCond)
submission$Foundation <- as.factor(submission$Foundation)
submission$BsmtQual <- as.factor(submission$BsmtQual)
submission$BsmtCond <- as.factor(submission$BsmtCond)
submission$BsmtExposure <- as.factor(submission$BsmtExposure)
submission$BsmtFinType1 <- as.factor(submission$BsmtFinType1)
submission$BsmtFinType2 <- as.factor(submission$BsmtFinType2)
submission$Heating <- as.factor(submission$Heating)
submission$HeatingQC <- as.factor(submission$HeatingQC)
submission$CentralAir <- as.factor(submission$CentralAir)
submission$Electrical <- as.factor(submission$Electrical)
submission$KitchenQual <- as.factor(submission$KitchenQual)
submission$Functional <- as.factor(submission$Functional)
submission$FireplaceQu <- as.factor(submission$FireplaceQu)
submission$GarageType <- as.factor(submission$GarageType)
submission$GarageFinish <- as.factor(submission$GarageFinish)
submission$GarageQual <- as.factor(submission$GarageQual)
submission$GarageCond <- as.factor(submission$GarageCond)
submission$PavedDrive <- as.factor(submission$PavedDrive)
submission$PoolQC <- as.factor(submission$PoolQC)
submission$Fence <- as.factor(submission$Fence)
submission$MiscFeature <- as.factor(submission$MiscFeature)
submission$SaleType <- as.factor(submission$SaleType)
submission$SaleCondition <- as.factor(submission$SaleCondition)

#Although these may be numeric, they should be factors
submission$MSSubClass <- as.factor(submission$MSSubClass)
submission$MoSold <- as.factor(submission$MoSold)

#Feature engineering
#Instead of having four variables which measure bathrooms, we create the TotBathrooms variable
submission$TotBathrooms <- submission$FullBath + (submission$HalfBath*0.5) + submission$BsmtFullBath + (submission$BsmtHalfBath*0.5)

#Setting variable of whether or not home has been remodeled
submission$Remod <- ifelse(submission$YearBuilt==submission$YearRemodAdd, 'No', 'Yes')
submission$Remod = as.factor(submission$Remod)

#Getting the age of the home as opposed to the year it was built
submission$Age <- as.numeric(submission$YrSold-submission$YearBuilt)

#Variable signifying whether or not the home is a new build
submission$IsNew <- ifelse(submission$YrSold==submission$YearBuilt, 'Yes', 'No')
submission$IsNew = as.factor(submission$IsNew)

#Drop the same variables we dropped in initial data
submission <- subset(submission, select = -c(YearBuilt, YearRemodAdd, FullBath, HalfBath, BsmtFullBath, BsmtHalfBath))
submission <- subset(submission, select = -c(GarageArea, TotRmsAbvGrd, TotalBsmtSF, GarageYrBlt))

#Convert submission data into a matrix
submission_matrix = data.matrix(submission)


#Make predictions using our xgboost model
preds = predict(boosted, submission)

#Create dataframe which combines ID labels with predictions
done <- data.frame(sub_ID_labels, preds)

done
