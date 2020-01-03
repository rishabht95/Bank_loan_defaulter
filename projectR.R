#remove all previous data
rm(list=ls(all=T))
#set current working directory
setwd("H:/Data Science/Project1")


#load libraries
x = c("ggplot2","DMwR","caret","randomForest","unbalanced","dummies","e1071",
      "Information","MASS","rpart","gbm","ROSE")


install.packages(x)
lapply(x,require,character.only=TRUE)
install.packages("dendextend")
library(dendextend)


install.packages('corrgram',dependencies = TRUE)
library(corrgram)
install.packages('C50',dependencies = TRUE)
library(C50)
install.packages('caret')
library(caret)
install.packages('randomForest')
library(randomForest)

#load data
loan_data=read.csv("H:/Data Science/Project1/bank-loan.csv")
unique=     unique(loan_data$othdebt)
length(unique)

#find missing values in each variable
missing_val=data.frame(apply(loan_data,2,function(x){sum(is.na(x))}))
#converting into proper row and column
missing_val$col=row.names(missing_val)
row.names(missing_val)=NULL

#changing variable names
names(missing_val)[1]="Missing Percentage"

#converting inti percentage
missing_val$`Missing Percentage`=(missing_val$`Missing Percentage`/nrow(loan_data))*100
#converting into descending order
missing_val=missing_val[order(-missing_val$`Missing Percentage`),]

#rearranging the columns
missing_val=missing_val[,c(2,1)]
missing_val=missing_val[order(-missing_val$`Missing Percentage`),]
View(missing_val)
#missing perc for def var=17.64706%

#write output ressult back into disk

write.csv(missing_val,"missing-perc.csv",row.names=F)
#above is the procedure how we convert our data in systematic form
#now how to put data in missing values columns using  central tendancy method and knn method
#there are missing values in our default column which is our dependent variable so we have to eliminate those rows
#because we can fill or predict the missing value for  for target variable


##eliminate the rows for what data is missing in target variable
loan_data=loan_data[!(is.na(loan_data$default)),]
###################OUTLIER ANALYSIS
#graphical method box plot
#first classify the numeric data since  above method can only be applied on numeric data

#convert categorical data into factor form
loan_data$ed=as.factor(loan_data$ed)
loan_data$default=as.factor(loan_data$default)

numeric_index=sapply(loan_data,is.numeric)
numeric_data=loan_data[,numeric_index]
cnames=colnames(numeric_data)

for(i in 1:length(cnames))
{
  assign(paste0("gn",i),ggplot(aes_string(y=(cnames[i]),x="default"),data=subset(loan_data))+
           stat_boxplot(geom="errorbar",width=.5)+
           geom_boxplot(outlier.colour="red",fill="grey",outlier.shape=18,
                        outlier.size=1,notch=FALSE)+
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="default")+
           ggtitle(paste("Box plot of defaulters for",cnames[i])))
}

#plotting plots together

gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
gridExtra::grid.arrange(gn3,gn4,gn6,gn7,ncol=2)




for(i in cnames){
  print(i)
  val=loan_data[,i][loan_data[,i] %in% boxplot.stats(loan_data[,i])$out]
  loan_data=loan_data[which(!loan_data[,i] %in% val),]
}


#################FEATURE SELECTION############################################
##Corelation plot for continuous variable
corrgram(loan_data[,cnames], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


install.packages("dendextend") #Adjust a tree's graphical parameters - the color, size, type, etc of its branches, nodes and labels
library(dendextend)


install.packages('corrgram',dependencies = TRUE)
library(corrgram)#The corrgram function produces a graphical display of a correlation matri

corrgram(loan_data[,numeric_index],order=F,
         upper.panel=panel.pie,text.panel=panel.txt,main="Correlation plot")
#chi squared test
#first select a ctegorical variable

factor_index=sapply(loan_data, is.factor)
factor_data=loan_data[,factor_index]
for(i in 1:1)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$default,factor_data[,i])))
}

##we will note remove any variable since in numerical variable analysis no one variable is highly related on another 
#in chi sq test p values is 0.0491
#we have to do feature scaling
##normality check to know which method to apply for feature scaling Selection

qqnorm(loan_data$income)
hist(loan_data$income)
#in above histogram w efind data is not uniformly distributed
#lets check for one more variable
hist(loan_data$address)
#in above we find data is left squeed
# so this data is not uniformly distributed ..therefore we will go with normalization
for(i in cnames){
  print(i)
  loan_data[,i]=(loan_data[,i]- min(loan_data[,i]))/(max(loan_data[,i]- min(loan_data[,i])))
}




#################algo implementation

##divide data into train and test using satisfied sampling method
set.seed(1234)
train.index=createDataPartition(loan_data$default,p=.80,list=FALSE)
train=loan_data[train.index,]
test=loan_data[-train.index,]


#######First apply regression analysis
#develop model
logit_model=glm(default~.,data=train,family = "binomial")

#Check summary
summary(logit_model)
#predict using ligistic regression
#response give output in probability format
logit_predictions=predict(logit_model,newdata=test,type="response")

#Convert prob
logit_predictions=ifelse(logit_predictions>0.5,1,0)


#evaluate the performance of classification model
confmatrix_RF=table(test$default,logit_predictions)

#FNR=0.6153
#FN=FN/FN+TP
#16/26
#Acc=TP+TN/(TP+TN+FP+FN)
#79/108

#ACCURACY=73.148
#FNR=0.6153

####We will apply some other models 

###############NAIVE BAYES
library(e1071)
#develop model
NB_model= naiveBayes(default~.,data=train)
#prediction on data
NB_prediction=predict(NB_model,test[,1:8],type='class')

##confusion matrix
conf_matrix=table(observed=test[,9],predicted=NB_prediction)
confusionMatrix(conf_matrix)
##accuracy=0.6667
##FNR=0.5384
#########################KNN

##apply knn
library(class)

knn_predictions=knn(train[,1:8],test[,1:8],train$default,k=1)


#confusion matrix
conf_matrix=table(knn_predictions,test$default)
confusionMatrix(conf_matrix)
##Accuracy=0.7037
###FNR=0.6071
##########################for k=3
knn_predictions=knn(train[,1:8],test[,1:8],train$default,k=3)


#confusion matrix
conf_matrix=table(knn_predictions,test$default)
confusionMatrix(conf_matrix)
##Accuracy=0.713
###FNR=0.6086
######################for  k =5
knn_predictions=knn(train[,1:8],test[,1:8],train$default,k=5)


#confusion matrix
conf_matrix=table(knn_predictions,test$default)
confusionMatrix(conf_matrix)
##Accuracy=0.7037
###FNR=0.65

######for k =3 accuracy is max and fnr is min so we go with k=3

###########APPLY DECISION TREE
###decision tree for classification
#develop model on training data
c50_model=C5.0(default ~.,train,trials=100,rules=TRUE)

#summary of c5 model
#here we get rules
summary(c50_model)
#let's predict for test cases
c50_predictions=predict(c50_model,test[,-9],type="class")

##Now applty this model on test data
#evaluate the performance of model using confusion metrix

confmatrix_c50=table(test$default,c50_predictions)
confusionMatrix(confmatrix_c50)

###Accuracy=0.6852
##FNR=0.73

#####go wit random forest
RF_model=randomForest(default~.,train,importance=TRUE,ntree=100,na.action=na.roughfix)
##Extract rules from RF_model

##First make a tree
install.packages("RRF")
library(RRF)


install.packages("inTrees")
library(inTrees)

treelist= RF2List(RF_model)

##extract rules
exec=extractRules(treelist,train[,-9])

##visualise some rules
exec[1:2,]

##make rules more readable
readablerules=presentRules(exec,colnames(train))
readablerules[1:2,]

#get rule metrics
ruleMetric=getRuleMetric(exec,train[,-9],train$default)
##evaluate few rules
ruleMetric[1:2,]

##predict test data using random forest model
rf_prediction=predict(RF_model,test[,-9])

##develop confusion matrix to ealuate performanece
confmatrix_RF=table(test$responded,rf_prediction)
confusionMatrix(confmatrix_RF)

###false negative rate
false_negative=FN/FN+TP
false_negative=16/26

#false_negative=0.6153
#accuracy=0.7315

##now check the model for n=500
RF_model=randomForest(default~.,train,importance=TRUE,ntree=500,na.action=na.roughfix)
rf_prediction=predict(RF_model,test[,-9])
confmatrix_RF=table(test$default,rf_prediction)
confusionMatrix(confmatrix_RF)

##accuracy=.7407
#false_negatove=0.5769

####now check the model for n=700
RF_model=randomForest(default~.,train,importance=TRUE,ntree=700,na.action=na.roughfix)
rf_prediction=predict(RF_model,test[,-9])
confmatrix_RF=table(test$default,rf_prediction)
confusionMatrix(confmatrix_RF)

##accuracy=.7315
#false_negatove=0.5769
##here accuracy rate decrease so n=500 our final tree



