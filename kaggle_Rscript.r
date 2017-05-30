library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)

# Load CSV files

data.train <- fread("~/train.csv", sep=",", na.strings = "NA")
data.test <- fread("~//test.csv", sep=",", na.strings = "NA")



# CV and seed
cvFoldsList <- createFolds(data.train$price_doc, k=5, list=TRUE, returnTrain=FALSE)
set.seed(2016)

# Transform target variable, so that we can use RMSE in XGBoost
data.train[,price_doc := log1p(as.integer(price_doc))]
data.test[,price_doc := "-1"]

# Combine data.tables
data <- rbind(data.train, data.test)

# Convert characters to factors/numeric

data[,":="(product_type=as.factor(product_type)
           ,sub_area=as.factor(sub_area)
           ,ecology=as.factor(ecology)
           ,thermal_power_plant_raion=ifelse(thermal_power_plant_raion=="no",0,1)
           ,incineration_raion=ifelse(incineration_raion=="no",0,1)
           ,oil_chemistry_raion=ifelse(oil_chemistry_raion=="no",0,1)
           ,radiation_raion=ifelse(radiation_raion=="no",0,1)
           ,railroad_terminal_raion=ifelse(railroad_terminal_raion=="no",0,1)
           ,big_market_raion=ifelse(big_market_raion=="no",0,1)
           ,nuclear_reactor_raion=ifelse(nuclear_reactor_raion=="no",0,1)
           ,detention_facility_raion=ifelse(detention_facility_raion=="no",0,1)
           ,culture_objects_top_25=ifelse(culture_objects_top_25=="no",0,1)
           ,water_1line=ifelse(water_1line=="no",0,1)
           ,big_road1_1line=ifelse(big_road1_1line=="no",0,1)
           ,railroad_1line=ifelse(railroad_1line=="no",0,1)
)]

# Date features
data[,timestamp := as.Date(timestamp)]
data[,":="(date_yday=yday(timestamp)
           ,date_month=month(timestamp)
           ,date_year=year(timestamp)
           ,date_week=week(timestamp)
           ,date_mday=mday(timestamp)
           ,date_wday=wday(timestamp)
)]
data[,":="(date_yearmonth=date_year*100+date_month
           ,date_yearweek=date_year*100+date_week
)]
data[,date_yearmonth_count := .N, keyby = date_yearmonth]
data[,date_yearweek_count := .N, keyby = date_yearweek]

# Count NA's
data[,na_count := rowSums(is.na(data))]

# Some relative features
data[,":="(rel_floor = floor/max_floor
           ,diff_floor = max_floor-floor
           ,rel_kitchen_sq = kitch_sq/full_sq
           ,rel_life_sq = life_sq/full_sq
           ,rel_kitchen_life = kitch_sq/life_sq
           ,rel_sq_per_floor = full_sq/floor
           ,diff_life_sq = full_sq-life_sq
           ,building_age = date_year - build_year
)]


# Load macro data

data.macro <- fread("~/macro.csv", sep=",", na.strings = "NA")
data.macro[,timestamp := as.Date(timestamp)]
data.macro <- sapply(data.macro,as.numeric) # Simply cast to numeric here, we can do feature engineering later
data <- merge(data, data.macro, by="timestamp", all.x=TRUE);gc()



# To sparse matrix

varnames <- setdiff(colnames(data), c("id", "price_doc", "timestamp"))
train_sparse <- Matrix(as.matrix(sapply(data[price_doc > -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
test_sparse <- Matrix(as.matrix(sapply(data[price_doc == -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
y_train <- data[price_doc > -1,price_doc]
test_ids <- data[price_doc == -1,id]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)





# Params for xgboost
param <- list(objective="reg:linear",
              eval_metric = "rmse",
              eta = .05,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)

 #Cross validation - determine CV scores & optimal amount of rounds

xgb_cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 1500,
                 maximize = FALSE,
                 prediction = TRUE,
                 folds = cvFoldsList,
                 print.every.n = 5,
                 early.stop.round = 100)

print(xgb_cv)
cv_log =xgb_cv$evaluation_log


rounds = which.min(xgb_cv$best_iteration)


xgb_cv$best_iteration

#rounds = which.min(xgb_cv$dt[, 'test_rmse_mean'])
#rounds <- which.min(xgb_cv$dt[, train.rmse.mean])
rounds <- 372


# Train model

xgb_model <- xgb.train(data = dtrain,
                       params = param,
                       watchlist = list(train = dtrain),
                       nrounds = rounds,
                       verbose = 1,
                       print.every.n = 5)

# Feature importance

names <- dimnames(train_sparse)[[2]]
importance_matrix <- xgb.importance(names,model=xgb_model)
xgb.plot.importance(importance_matrix[1:50,])

# Predict and output csv

preds <- predict(xgb_model,dtest)
preds <- expm1(preds)



write.table(data.table(id=test_ids, price_doc=preds), 
            "script_submission.csv", sep=",", dec=".", quote=FALSE, 
            row.names=FALSE)


#Build the feature importance data.table
#In the code below, sparse_matrix@Dimnames[[2]] represents the column names of 
#the sparse matrix. These names are the original values of the features 
#(remember, each binary column == one value of one categorical feature).
importance <- xgb.importance(feature_names = train_sparse@Dimnames[[2]], model = xgb_model)
head(importance)
top_20 =importance[1:20,]



#The column Gain provide the information we are looking for.
#Gain is the improvement in accuracy brought by a feature to the branches it is on.


























