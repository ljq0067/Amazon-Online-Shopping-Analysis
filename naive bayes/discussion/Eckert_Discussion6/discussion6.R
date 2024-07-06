################################
# Haley Eckert
# Fall, 2021
# Discussion 6
# Naive Bayes on Record Data
################################

# Libraries
library(e1071)
library(ggplot2)
library(caret)
library(cvms)

####################################################
### Read in Data, Clean, & View
####################################################

# Set working directory
MyPath = "/Users/haleyroberts/Desktop/501_Intro/Discussions/discussion6/"
setwd(MyPath)

#Read in data
dataFilename = "data.csv"
dataDF <- read.csv(dataFilename, stringsAsFactors=TRUE)
head(dataDF)

# Check initial data types
str(dataDF)

# Change Num_Siblings to type Factor
dataDF$num_siblings <- as.factor(dataDF$num_siblings)

# Check updated data types
str(dataDF)

####################################################
### Split data into Train & Test sets
### by random sampling without replacement
####################################################

# Count rows
(DataSize=nrow(dataDF))
# Find size of training set (to be 75% of data)
(TrainingSet_Size<-floor(DataSize*(3/4)))
# Find size of testing set (to be 25% of data)
(TestSet_Size <- DataSize - TrainingSet_Size)

# Set seed
set.seed(123)

# Sample for row numbers for training set
(MyTrainSample <- sample(nrow(dataDF), TrainingSet_Size, replace=FALSE))
# Select training set from data frame using sample of row numbers
(MyTrainingSET <- dataDF[MyTrainSample,])
# View counts of each label in training set
table(MyTrainingSET$highest_education)

# Select testing set from dataframe using data not in training set
(MyTestSET <- dataDF[-MyTrainSample,])
# View counts of each label in testing set
table(MyTestSET$highest_education)  # At least 1 of each category, so good

# Remove labels from test set and store in variable
(TestKnownLabels <- MyTestSET$highest_education)
(MyTestSET <- MyTestSET[ , -which(names(MyTestSET) %in% c("highest_education"))])

# Remove labels from test set and store in variable
(TrainKnownLabels <- MyTrainingSET$highest_education)
(MyTrainingSET <- MyTrainingSET[ , -which(names(MyTrainingSET) %in% c("highest_education"))])

####################################################
### Run Naive Bayes
####################################################

# Run naive bayes model
(NB_e1071_2<-naiveBayes(MyTrainingSET, 
                        TrainKnownLabels, 
                        laplace = 1))

# Predict test labels using model
NB_e1071_Pred <- predict(NB_e1071_2, MyTestSET)

# View confusion matrix
(confMat <- table(NB_e1071_Pred,TestKnownLabels))

# Calculate accuracy
(accuracy <- (confMat[1, 1] + confMat[2, 2] + confMat[3, 3] + confMat[4, 4])/sum(colSums(confMat)))

# Plot Confusion Matrix
(cm1 <- caret::confusionMatrix(NB_e1071_Pred, TestKnownLabels, positive="true"))
cmDF1 <- as.data.frame(cm1$table)
plot_confusion_matrix(cmDF1, 
                      target_col = "Reference", 
                      prediction_col = "Prediction",
                      counts_col = "Freq",
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE,
                      rm_zero_percentages = FALSE,
                      rm_zero_text = FALSE,
                      add_zero_shading = TRUE,
                      counts_on_top = TRUE)
# Find accuracy of matrix
#(accuracy1 <- cm1$overall[1])
