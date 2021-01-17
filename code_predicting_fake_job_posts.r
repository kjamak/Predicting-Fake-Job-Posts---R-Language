## Step 1:  In this step we will import all of the packages  that will be used during our project ---

library(tidyverse) 
library(dslabs)
library(dplyr)
library(tinytex)
library(gridExtra)
library(grid) 
library(caret)  
library(lubridate)  
library(data.table)
library(tidytext)
library(stopwords)  
library(tm)      
library(SnowballC) 
library(wordcloud)
library(e1071)    
library(kernlab)
library(readr)  




## Step 2: Load and Exploration of the Data ---

allJobPosts <- read.csv("fake_job_postings.csv")

# Let's do some exploration of data

str(allJobPosts)

# examine the fraudulent variable more carefully

str(allJobPosts$fraudulent)

# lets see number of true and fake job posts, class variable

table(allJobPosts$fraudulent)

#another way more informational in %

round(prop.table(table(allJobPosts$fraudulent)) * 100, digits = 2)

#let's do exploration of other collumns to better learn the data

table(allJobPosts$employment_type)
table(allJobPosts$required_experience)


#check for NAs in our Decision class

sum(is.na(allJobPosts$fraudulent))

# let's  drop id section from fake_job_posts

allJobPosts <- allJobPosts[-1]

# Let's plot also for better data representation for Report

ggplot(allJobPosts, aes(fraudulent, fill = fraudulent))+
  geom_bar()+
  theme_classic()

# Create the corpus object based on description of the job post
# Build a corpus using the text mining (tm) package

corpObject <- Corpus(VectorSource(allJobPosts$description))

# Examine the  corpus Object
print(corpObject)

# Remove punctuation from our corpus object
corpObject <- tm_map(corpObject, removePunctuation)
# Also remove stop words with already built in funcition
corpObject <- tm_map(corpObject, removeWords, stopwords(kind = "en"))

# Perform stemming on our corpus object just like in spam or ham example from class
corpObject <- tm_map(corpObject, stemDocument)

# Next step is to create the list of frequently used words with documentTermMattrix
freqWords <- DocumentTermMatrix(corpObject)
# Removing sparse data
spData <- removeSparseTerms(freqWords, 0.995)
# Converting to dataframe so we can do further exploration and analysis
spData_df <- as.data.frame(as.matrix(spData))
# Assigning column names
colnames(spData_df) <- make.names(colnames(spData_df))
# Adding the dependent variable
spData_df$fraudulent <- allJobPosts$fraudulent 
# Removing duplicate column names
colnames(spData_df) <- make.unique(colnames(spData_df), sep = "_")


## Step 3: Training a model on the data ---


# Let's set seed with sample kind of rounding
set.seed(123, sample.kind = "Rounding")
# Lets create test data that will be 10%
testData <- createDataPartition(y = spData_df$fraudulent, times = 1, p = 0.1, list= FALSE)
# Create train data that will be used for training our algorithm 90% of data portion
trainData <- spData_df[-testData, ]
# Test data 10% portion 
testValidation <- spData_df[testData, ]

# create labels for training and test data

trainData$fraudulent = as.factor(trainData$fraudulent)
testValidation$fraudulent = as.factor(testValidation$fraudulent)


# Creating train control for knn 
trainControl <- trainControl(method = "cv", verboseIter = TRUE, number = 5)

# Training the KNN algorithm and tuning for optimized k
knnAlgorithm <- train(fraudulent ~ .,
                      data = trainData, method = "knn", preProcess = c("center","scale"),
                      trControl = trainControl , tuneGrid = expand.grid(k = c(5)))


## Step 4: Evaluating model performance ----

#  Predicting fraud job postings
knnPredict <- predict(knnAlgorithm,newdata = testValidation )
#  Confusion matrix for our knn Algorithm
matrixKnn <- confusionMatrix(knnPredict, testValidation$fraudulent )

matrixKnn

# Accuracy for knn is near 97%

#-----------------------------------------------------------

## traingn the algorithm using Support Vector Machines (SVM)

# creating train control for our svm algorithm

trainControl <- trainControl(method = "cv", verboseIter = TRUE, number = 5)


# training the Support Vector Machines (SVM) algorithm

svmAlgorithm <- train(fraudulent ~ .,data = trainData, 
                      method = "svmLinear", preProcess = c("center","scale"),
                      tuneGrid = expand.grid(C = c(0.01)), trControl = trainControl)


## Evaluating svm model performance ---

#Prediction with our SVM Model
svmPredict <- predict(svmAlgorithm, newdata = testValidation)

#Confusion Matrix for our SVM Model
matrixSvm <- confusionMatrix(svmPredict, testValidation$fraudulent)

matrixSvm



## Step 5: Improving model performance ---- 

# Let's see will model for knn improve if we increase k to  tune up to 10


# Creating train control for knn up to 10 but we will stick with 5 folds
trainControl <- trainControl(method = "cv", verboseIter = TRUE, number = 5)

# Training the KNN algorithm and tuning for optimized k
knnAlgorithm <- train(fraudulent ~ .,
                      data = trainData, method = "knn", preProcess = c("center","scale"),
                      trControl = trainControl , tuneGrid = expand.grid(k = c(10)))


## Evaluating try of improved model performance ----

#  Predicting fraud job postings
knnPredict <- predict(knnAlgorithm,newdata = testValidation )
#  Confusion matrix for our knn Algorithm
matrixKnn <- confusionMatrix(knnPredict, testValidation$fraudulent )

matrixKnn
