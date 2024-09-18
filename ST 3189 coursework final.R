# 200552574 Coursework ------------------------------------

library(tidyverse)
library(ggpubr)
library(caTools) 
library(rpart)
library(rpart.plot)
library(ggplot2)

#Classification and Unsupervised Learning----------------------------

heart1 <- read.csv("heart.csv")
which(is.na(heart1))
sum(is.na(heart1))

heart <- heart1

## There  is no NA values in the data.
summary(heart)
## checke for zero values
boxplot(heart$Cholesterol , heart$RestingBP , heart$MaxHR ,heart$Oldpeak ,
        main = "Boxplot of Numerical Values",
        names = c("Cholesterol","RestingBP", "MaxHR", "OldPeak"))
## replace zero with median
median(heart$Cholesterol) #median 223
median(heart$RestingBP) #median 130

heart$Cholesterol <- ifelse(heart$Cholesterol == 0, median(heart$Cholesterol), heart$Cholesterol)
heart$RestingBP <- ifelse(heart$RestingBP == 0, median(heart$RestingBP), heart$RestingBP)

boxplot(heart$Cholesterol , heart$RestingBP , 
        main = "Boxplot of Cholesterol and Resting BP",
        names = c("Cholesterol","RestingBP"))
# changing the values of the columns.
heart$Sex <- ifelse(heart$Sex == 'M', 0, 1)
heart$ChestPainType <- ifelse(heart$ChestPainType == 'ATA', 0, ifelse(heart$ChestPainType == 'NAP', 1, ifelse(heart$ChestPainType == 'ASY', 2, 3)))
heart$RestingECG <- ifelse(heart$RestingECG == 'Normal', 1, ifelse(heart$RestingECG == 'ST', 2, 3))
heart$ExerciseAngina <- ifelse(heart$ExerciseAngina == 'N', 1, 2)
heart$ST_Slope <- ifelse(heart$ST_Slope == 'Up', 0, ifelse(heart$ST_Slope == 'Flat', 1, 2))

heart.dum <- heart
heart.dum$HeartDisease <- as.factor(heart.dum$HeartDisease) 
heart.dum$Sex <- as.factor(heart.dum$Sex)
heart.dum$ChestPainType <- as.factor(heart.dum$ChestPainType)
heart.dum$FastingBS <-  as.factor(heart.dum$FastingBS)
heart.dum$RestingECG <-  as.factor(heart.dum$RestingECG)
heart.dum$ExerciseAngina <-  as.factor(heart.dum$ExerciseAngina)
heart.dum$ST_Slope <-  as.factor(heart.dum$ST_Slope)

## QN: Classification task -----------------------------------------------------
heart2 <- heart.dum

library(caret)
library(caTools)

### Logistic Regression -------------------------------------------------
#Train-Test Split
set.seed(2020)
heart_sampling <- createDataPartition(heart2$HeartDisease, p=0.70, list = FALSE)
trainLR <- heart2[heart_sampling,]
trainLR_labels <- heart2$HeartDisease[heart_sampling]
testLR <- heart2[-heart_sampling,]
testLR_label <- heart2$HeartDisease[-heart_sampling]

heartLR_model <- glm(HeartDisease ~ . , data= trainLR, family = binomial("logit"))
summary(heartLR_model)

trainLR_predictions <- predict(heartLR_model, newdata = trainLR, type = "response")
trainLR_class_predictions <- as.numeric(trainLR_predictions > 0.5)
mean(trainLR_class_predictions == trainLR$HeartDisease)

testLR_predictions <- predict(heartLR_model, newdata = testLR, type = "response")
testLR_class_predictions = as.numeric(testLR_predictions > 0.5)
mean(testLR_class_predictions == testLR$HeartDisease)

# Classification Matrix
(confusion_matrix <- table(predicted = testLR_class_predictions, actual = testLR$HeartDisease))
summary(confusion_matrix)
(precision <- confusion_matrix[2, 2] / sum(confusion_matrix[2,]))
(recall <- confusion_matrix[2, 2] / sum(confusion_matrix[,2]))
(f = 2 * precision * recall / (precision + recall))
(specificity <- confusion_matrix[1, 1]/sum(confusion_matrix[1,]))

confusiondf <- as.data.frame(as.table(confusion_matrix))
ggplot(confusiondf, aes(x = actual, y = predicted, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(x = "Actual", y = "Predicted", fill = "Frequency") +
  theme_minimal()

### Support Vector Machine (SVM) ----------------------------------------
# Support Vector Machine factor all the variables other than heart disease into new data.
heart.dum1 <- heart
heart.dum1$Sex <- as.factor(heart.dum1$Sex)
heart.dum1$ChestPainType <- as.factor(heart.dum1$ChestPainType)
heart.dum1$FastingBS <-  as.factor(heart.dum1$FastingBS)
heart.dum1$RestingECG <-  as.factor(heart.dum1$RestingECG)
heart.dum1$ExerciseAngina <-  as.factor(heart.dum1$ExerciseAngina)
heart.dum1$ST_Slope <-  as.factor(heart.dum1$ST_Slope)


dummies <- dummyVars(HeartDisease ~ ., data = heart.dum1)
heart3 <- data.frame(predict(dummies, newdata = heart.dum1), HeartDisease = factor((heart.dum1$HeartDisease)))
dim(heart3)
set.seed(2020)
svm_sampling <- createDataPartition(heart3$HeartDisease, p = 0.70, list = FALSE)
svm_train <- heart3[svm_sampling,]
svm_test <- heart3[-svm_sampling,]

library(e1071)
set.seed(2020)
svmradial_tune <- tune(svm,HeartDisease ~ ., data = svm_train, kernel = "radial", ranges = list(cost = c(0.01, 0.1, 1, 10, 100), gamma = c(0.01, 0.05, 0.1, 0.5, 1)))
svmradial_tune$best.parameters # Cost = 10, gamma = 0.01
svmradial_tune$best.performance # 0.1353846
100 - (0.1353846*100) #  Training Accuracy = 86%

svm_model <- svmradial_tune$best.model
svmtest_predictions <- predict(svm_model, svm_test[,1:21])
mean(svmtest_predictions == svm_test[,22])
(svm_confusion <- table(predicted = svmtest_predictions, actual = svm_test[,22]))

svmconfusiondf <- as.data.frame(as.table(svm_confusion))
(precisionsvm <- svm_confusion[2, 2] / sum(svm_confusion[2,]))
(recallsvm <- svm_confusion[2, 2] / sum(svm_confusion[,2]))
(fsvm = 2 * precisionsvm * recallsvm / (precisionsvm + recallsvm))
(specificitysvm <- svm_confusion[1, 1]/sum(svm_confusion[1,]))

ggplot(svmconfusiondf, aes(x = actual, y = predicted, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(x = "Actual", y = "Predicted", fill = "Frequency") +
  theme_minimal()


### K-nearest neigbhour -------------------------------------------------------- 
heartk <- heart
heartk$HeartDisease <-ifelse(heartk$HeartDisease ==0,'Present','Absent')
heartk$HeartDisease <- factor(heartk$HeartDisease, levels = c("Present", "Absent"))
set.seed(2020)
knn_sampling <- createDataPartition(heartk$HeartDisease, p = 0.7, list = FALSE)
knn_train <- heartk[knn_sampling,]
knn_test <- heartk[-knn_sampling,]

# Grid values to test in cross-validation

knngrid <- expand.grid(k = c(1:20))
fitcontrol <- trainControl(method = "cv",
                           classProbs = TRUE,)

knnfit <- train(HeartDisease ~., data = knn_train, method = "knn", trControl = fitcontrol, tuneGrid = knngrid, metric = "Accuracy")
knnfit
plot(knnfit)
str(knnfit,1)
knnfit$finalModel

knnpred_class <- predict(knnfit, knn_test, 'raw')
probs <- predict(knnfit, knn_test, 'prob')

knntest_score <- cbind(knn_test, knnpred_class, probs)
glimpse(knntest_score)

library(yardstick)
knnFit.conf_mat <- knntest_score %>% 
  conf_mat(truth=HeartDisease , knnpred_class)

knnconfusiondf <- as.data.frame(as.table(knnFit.conf_mat[["table"]]))
ggplot(knnconfusiondf, aes(x = Truth, y = Prediction, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(x = "Actual", y = "Predicted", fill = "Frequency") +
  theme_minimal()

## QN: Unsupervised Learning --------------------------------------------

### PCA ------------------------------------------------------------

heartp <- heart
heartpca <- prcomp(heartp, scale = T)
summary(heartpca)
## the first 6 pc explains abput 72% of the variance in the data.

heartpca$rotation
heartpca$sdev^2 / sum(heartpca$sdev^2)

var_explained = heartpca$sdev^2 / sum(heartpca$sdev^2)
qplot(c(1:12), var_explained) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)

heart.scaled <- scale(heart)

### K-Means Clustering ------------------------------------------------------------
set.seed(2020)
k2 <- kmeans(heart.scaled, centers=2)
summary(k2)
k2results <- data.frame(heart$Sex, heart$HeartDisease, k2$cluster)
cluster1 <- subset(k2results, k2$cluster==1)
cluster2 <- subset(k2results, k2$cluster==2)
cluster1$heart.Sex <- factor(cluster1$heart.Sex)
cluster2$heart.Sex <- factor(cluster2$heart.Sex)

summary(cluster1$heart.HeartDisease)
summary(cluster2$heart.HeartDisease)
## cluster 1 has more people with heart disease than cluster 2.

round(prop.table(table(cluster1$heart.Sex)),2)
round(prop.table(table(cluster2$heart.Sex)),2)
## 90% in cluster 1 are males, 65% in cluster 2 are males.

#Goodness of fit test
# Is cluster 1 statistically the same as cluster 2 in terms of sex?
M <- as.matrix(table(cluster1$heart.Sex))
p.null <- as.vector(prop.table(table(cluster2$heart.Sex)))
chisq.test(M, p=p.null)
# This shows that the distribution of the sexes in cluster 1 is significantly different from cluster 2
# this does not proof if the gender affects the possibility of a person getting heart disease.

### Hierarchical Clustering -----------------------------------------------------------------------
hc.average =hclust(dist(heart.scaled), method="average")

plot(hc.average , main = "Average Linkage", xlab="", sub="", cex =.9)
sum(cutree(hc.average , 3)==2) ## 8 cases in second cluster
## Average Linkage fails to provide sufficient sample size.

hc.complete =hclust(dist(heart.scaled), method ="complete")
plot(hc.complete , main ="Complete Linkage", xlab="", sub="", cex =.9)
sum(cutree(hc.complete, 3)==2) ## 210 cases in second cluster
sum(cutree(hc.complete, 3)==3)
hc.cluster1 <- subset(k2results, cutree(hc.complete, 3)==1)
hc.cluster2 <- subset(k2results, cutree(hc.complete, 3)==2)
hc.cluster3 <- subset(k2results, cutree(hc.complete, 3)==3)

hc.cluster1$heart.Sex <- factor(hc.cluster1$heart.Sex)
hc.cluster2$heart.Sex <- factor(hc.cluster2$heart.Sex)
hc.cluster3$heart.Sex <- factor(hc.cluster3$heart.Sex)

summary(hc.cluster1$heart.HeartDisease)
summary(hc.cluster2$heart.HeartDisease)
summary(hc.cluster3$heart.HeartDisease)
## cluster 2 has the most people with heart disease followed by cluster 3 and cluster 3 has the least.

round(prop.table(table(hc.cluster1$heart.Sex)),2)
round(prop.table(table(hc.cluster2$heart.Sex)),2)
round(prop.table(table(hc.cluster3$heart.Sex)),2)
## 86% in cluster 3 are male, 81% in cluster 2 are male, 78% in cluster 1 are male.

# Goodness of Fit Test
# is hc.cluster 2 statistically same as hc.cluster 1 in terms of sex.
M1 <- as.matrix(table(hc.cluster2$heart.Sex))
p.null1 <- as.vector(prop.table(table(hc.cluster1$heart.Sex)))
chisq.test(M1, p=p.null1)
# cluster 1 and cluster 2 is statistically similar

# is hc.cluster 2 statistically same as hc.cluster 3 in terms of sex.
M1 <- as.matrix(table(hc.cluster2$heart.Sex))
p.null2 <- as.vector(prop.table(table(hc.cluster3$heart.Sex)))
chisq.test(M1, p=p.null2)
## We cannot confidently conclude that they statistically different but there is some indication that they sare different

# is hc.cluster 1 statistically same as hc.cluster 3 in terms of sex.
M2 <- as.matrix(table(hc.cluster1$heart.Sex))
p.null2 <- as.vector(prop.table(table(hc.cluster3$heart.Sex)))
chisq.test(M2, p=p.null2)
## They are statistically different.
## we conclude that gender is important to predict whether the person has heart disease.




# Regression ----------------------------------------------------------

## QN: Regression Task -------------------------------------------------------
property <- read.csv("Housing_Price_Data.csv")

sum(is.na(property))
## there is no NA values.

summary(property)

property$mainroad <- ifelse(property$mainroad == 'no', 0, 1)
property$guestroom <- ifelse(property$guestroom == 'no', 0, 1)
property$basement <- ifelse(property$basement == 'no', 0, 1)
property$hotwaterheating <- ifelse(property$hotwaterheating == 'no', 0, 1)
property$airconditioning <- ifelse(property$airconditioning == 'no', 0, 1)
property$prefarea <- ifelse(property$prefarea == 'no', 0, 1)
property$furnishingstatus <- ifelse(property$furnishingstatus == 'furnished', 0, ifelse(property$furnishingstatus == 'semi-furnished', 1, 2))

str(property)

# Train-Test split

set.seed(2020)
trainp <- createDataPartition(property$price, p = 0.7, list = FALSE)
trainset <- property[trainp,]
testset <- property[-trainp,]

### Linear Regression -------------------------------------------------------------------
m.linear <- lm(price ~ ., data = trainset)
summary(m.linear)    ## R^2 is 0.68.
model <- ("Linear Reg")
RMSE.train <- round(sqrt(mean((trainset$price - predict(m.linear))^2)))   ## 1055239
RMSE.test <- round(sqrt(mean((testset$price - predict(m.linear, newdata = testset))^2))) ## 1077030  

### Random Forest (RF) ------------------------------------------------------------
# RF at default settings of ntree & mtry 
library(randomForest)
m.rf <- randomForest(price ~ . , data = trainset)
m.rf  ## % Var explained : 63.8%

# OOB RMSE
sqrt(m.rf$mse[m.rf$ntree]) # 1146463

plot(m.rf)
## confirms that the eroor has stabalised before 500 trees.

model <- c(model, "Random Forest")

RMSE.train <- c(RMSE.train, round(sqrt(mean((trainset$price - predict(m.rf, newdata = trainset))^2)))) #630813

RMSE.test <- c(RMSE.test, round(sqrt(mean((testset$price - predict(m.rf, newdata = testset))^2)))) #1054046

###CART -----------------------------------------
# Optimal CART using 1SE in maximal CART
m.cart <- rpart(price ~ ., method = "anova", cp = 0, data = trainset)

m.cart$variable.importance
## without pruning the tree it shows that area, bathrooms, stories and parking are the more significant factor affecting the price
round(100*m.cart$variable.importance/sum(m.cart$variable.importance))
## area is the most significant factor affecting the price.
rpart.plot(m.cart, type = 1, extra = 1)

# Compute min CVerror + 1SE in maximal CART
CVerror.cap <- m.cart$cptable[which.min(m.cart$cptable[,"xerror"]), "xerror"] + m.cart$cptable[which.min(m.cart$cptable[,"xerror"]), "xstd"]
## 0.52555454056988

#Find the optimal CP region whose CV error is just below CVerror in maximal tree.
i <- 1; j <- 4
while(m.cart$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}
cp.opt = ifelse(i > 1, sqrt(m.cart$cptable[i,1] * m.cart$cptable[i-1,1]), 1)
printcp(m.cart)
plotcp(m.cart)

## From the code and tree we can see that i = 8 is optimal.

# Prune to get 1SE optimal CART via cp.opt 
m.cart.1se <- prune(m.cart, cp = cp.opt)
rpart.plot(m.cart.1se, type = 1, extra = 1)

m.cart.1se$variable.importance
## Areas, Bathrooms and Stories are significant variables.

model <- c(model, "CART 1SE")

## Regression Result ---------------------------
RMSE.train <- c(RMSE.train, round(sqrt(mean((trainset$price - predict(m.cart.1se))^2)))) ## 1213428

RMSE.test <- c(RMSE.test, round(sqrt(mean((testset$price - predict(m.cart.1se, newdata = testset))^2)))) ## 1390754

regresults <- data.frame(model, RMSE.train, RMSE.test)
view(regresults)
## Random Forest has the lowest test set error.