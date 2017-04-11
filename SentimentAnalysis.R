#Including Library required for the project, major one is text mining library 'tm'.

library(tm)
library(SnowballC)
library(gtools)
library(klaR)
library(caret)
library(e1071)
library(kknn)
library(ROCR)
library(kernlab)
library(Deducer)
library(randomForest)
library(foreign)


genarrayperf<-function(cvs)
{
  folds<-unique(cvs$Resample) 
  result<-c(1:length(folds))
  cont<-1
  for (i in folds)
  {
    cm<-confusionMatrix(cvs[cvs$Resample==i,'obs'],cvs[cvs$Resample==i,'pred'])
    result[cont]<-cm$overall[1] 
    cont<-cont+1
  }
  return(result)
}

##################################################PreProcessing data(Positive and Neative Reviews)###########################################

dataSource <- DirSource(directory = "D:/subject/spring17/NLP/project/dataSet/pos",mode ="text")  
amazonCorpusPos <- Corpus(dataSource, readerControl=list(reader=readPlain))
 
dataSource <- DirSource("D:/subject/spring17/NLP/project/dataSet/neg",mode="text")  
amazonCorpusNeg <- Corpus(dataSource, readerControl=list(reader=readPlain))  
 
#Stop word, Number and whitespace removal
amazonCorpusProcessedNeg <- tm_map(amazonCorpusNeg, removeNumbers)
amazonCorpusProcessedNeg <- tm_map(amazonCorpusProcessedNeg , stripWhitespace)
amazonCorpusProcessedNeg <- tm_map(amazonCorpusProcessedNeg, removeWords, stopwords("english"))
#amazonCorpusProcessedNeg <- tm_map(amazonCorpusProcessedNeg, removePunctuation)
#amazonCorpusProcessedNeg <- tm_map(amazonCorpusProcessedNeg, tolower)
#amazonCorpusProcessedNeg <- tm_map(amazonCorpusProcessedNeg, stemDocument, language = "english")
 


#Stop word, Number and whitespace removal
amazonCorpusProcessed <- tm_map(amazonCorpusPos, removeNumbers)
amazonCorpusProcessed <- tm_map(amazonCorpusProcessed , stripWhitespace)
amazonCorpusProcessed <- tm_map(amazonCorpusProcessed, removeWords, stopwords("english"))
#amazonCorpusProcessed <- tm_map(amazonCorpusProcessed, removePunctuation)
#amazonCorpusProcessed <- tm_map(amazonCorpusProcessed, tolower)
#amazonCorpusProcessed <- tm_map(amazonCorpusProcessed, stemDocument, language = "english")

################################### Document Term Matrix creation and Respective Data Frame creation #######################################
          
#Unigram 
UnigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 1), paste, collapse = " "), use.names = FALSE)

#positive
amazonDtmPositiveUnigram <-DocumentTermMatrix(amazonCorpusProcessed) 
amazonDtmPositiveUnigram <- removeSparseTerms(amazonDtmPositiveUnigram, 0.75)  

amazonDataFramePositive <- as.data.frame(inspect(amazonDtmPositiveUnigram))
rownames(amazonDataFramePositive) <- 1:nrow(amazonDataFramePositive)
PosNeg <- as.factor(rep(1,nrow(amazonDataFramePositive)))
amazonDataFramePositive<- cbind(amazonDataFramePositive,PosNeg)

#negative
amazonDtmNegativeUnigram <-DocumentTermMatrix(amazonCorpusProcessedNeg) 
amazonDtmNegativeUnigram <- removeSparseTerms(amazonDtmNegativeUnigram, 0.75)   

amazonDataFrameNegative <- as.data.frame(inspect(amazonDtmNegativeUnigram))
rownames(amazonDataFrameNegative) <- 1:nrow(amazonDataFrameNegative)
PosNeg <- as.factor(rep(0,nrow(amazonDataFrameNegative)))
amazonDataFrameNegative<- cbind(amazonDataFrameNegative,PosNeg)
 

#BiGram
#Positive 
BigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

amazonTermDocFreqPositiveBigram <- TermDocumentMatrix(amazonCorpusProcessed, control = list(tokenize = BigramTokenizer))
amazonDtmPositiveBigram <- DocumentTermMatrix(amazonCorpusProcessed, control = list(tokenize = BigramTokenizer,minWordLength = 2,minDocFreq = 3))
amazonDtmPositiveBigram <- removeSparseTerms(amazonDtmPositiveBigram, 0.85)

amazonDataFramePositiveBigram <- as.data.frame(inspect(amazonDtmPositiveBigram))
rownames(amazonDataFramePositiveBigram) <- 1:nrow(amazonDataFramePositiveBigram)
PosNeg <- as.factor(rep(1,nrow(amazonDataFramePositiveBigram)))
amazonDataFramePositiveBigram <- cbind(amazonDataFramePositiveBigram,PosNeg)

#Negative
BigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

amazonTermDocFreqNegativeBigram <- TermDocumentMatrix(amazonCorpusProcessedNeg, control = list(tokenize = BigramTokenizer))
amazonDtmNegativeBigram <- DocumentTermMatrix(amazonCorpusProcessedNeg, control = list(tokenize = BigramTokenizer,minWordLength = 2,minDocFreq = 3))
amazonDtmNegativeBigram <- removeSparseTerms(amazonDtmNegativeBigram, 0.85)

amazonDataFrameNegativeBigram <- as.data.frame(inspect(amazonDtmNegativeBigram))
rownames(amazonDataFrameNegativeBigram) <- 1:nrow(amazonDataFrameNegativeBigram)
PosNeg <- as.factor(rep(0,nrow(amazonDataFrameNegativeBigram)))
amazonDataFrameNegativeBigram <- cbind(amazonDataFrameNegativeBigram,PosNeg)

#Trigram 

#Positive
TrigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)
amazonTermDocFreqPositiveTrigram <- TermDocumentMatrix(amazonCorpusProcessed, control = list(tokenize = TrigramTokenizer))
amazonDtmPositiveTrigram <- DocumentTermMatrix(amazonCorpusProcessed, control = list(tokenize = TrigramTokenizer,minWordLength = 2,minDocFreq = 3))
amazonDtmPositiveTrigram <- removeSparseTerms(amazonDtmPositiveTrigram, 0.85)  

amazonDataFramePositiveTrigram <- as.data.frame(inspect(amazonDtmPositiveTrigram))
rownames(amazonDataFramePositiveTrigram) <- 1:nrow(amazonDataFramePositiveTrigram)
PosNeg <- as.factor(rep(1,nrow(amazonDataFramePositiveTrigram)))
amazonDataFramePositiveTrigram <- cbind(amazonDataFramePositiveTrigram,PosNeg)


#Negative 

TrigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)
amazonTermDocFreqNegativeTrigram <- TermDocumentMatrix(amazonCorpusProcessedNeg, control = list(tokenize = TrigramTokenizer))
amazonDtmNegativeTrigram <- DocumentTermMatrix(amazonCorpusProcessedNeg, control = list(tokenize = TrigramTokenizer,minWordLength = 2,minDocFreq = 3))
amazonDtmNegativeTrigram <- removeSparseTerms(amazonDtmNegativeTrigram, 0.85)  

amazonDataFrameNegativeTrigram <- as.data.frame(inspect(amazonDtmNegativeTrigram))
rownames(amazonDataFrameNegativeTrigram) <- 1:nrow(amazonDataFrameNegativeTrigram)
PosNeg <- as.factor(rep(0,nrow(amazonDataFrameNegativeTrigram)))
amazonDataFrameNegativeTrigram <- cbind(amazonDataFrameNegativeTrigram,PosNeg)



####################Binding of Positive and Negative Reviews for UNI , BI , TRI GRAM########################

#unigram

amazonDataFrameCombine <- smartbind(amazonDataFramePositive,amazonDataFrameNegative,fill=0)
#amazonDataFrameCombine <- amazonDataFrameCombine[c("PosNeg",names(amazonDataFrameCombine)[-215])]

#Bigram 

amazonDataFrameBICombine <- smartbind(amazonDataFramePositiveBigram,amazonDataFrameNegativeBigram,fill=0)

#Trigram

amazonDataFrameTRICombine <- smartbind(amazonDataFramePositiveTrigram,amazonDataFrameNegativeTrigram,fill=0)

#Unigram,BiGram,TriGram combined

amazonDataFrameUNIBITRICombine <- smartbind(cbind(amazonDataFramePositive,amazonDataFramePositiveBigram,amazonDataFramePositiveTrigram),cbind(amazonDataFrameNegative,amazonDataFrameNegativeBigram,amazonDataFrameNegativeTrigram),fill=0)



###################################### Train and Test Set for all Algorithms ###############################

## Split as train and set
set.seed(20)

## Unigram train and test data
randIndex <- rbinom(nrow(amazonDataFrameCombine),1,0.5)
trainingData <- amazonDataFrameCombine[randIndex == 1,]
trainingData
testingData <- amazonDataFrameCombine[randIndex == 0,]
testingData

## Bigram train and test data
randIndex <- rbinom(nrow(amazonDataFrameBICombine),1,0.5)
trainingDataBigram <- amazonDataFrameBICombine[randIndex == 1,]
testingDataBigram <- amazonDataFrameBICombine[randIndex == 0,]


## Trigram train and test data
randIndex <- rbinom(nrow(amazonDataFrameTRICombine),1,0.5)
trainingDataTrigram <- amazonDataFrameTRICombine[randIndex == 1,]
testingDataTrigram <- amazonDataFrameTRICombine[randIndex == 0,]

## Unigram and Bigram train and test data
randIndex <- rbinom(nrow(amazonDataFrameUNIBITRICombine),1,0.5)
trainingDataUniBi <- amazonDataFrameUNIBITRICombine[randIndex == 1,]
testingDataUniBi <- amazonDataFrameUNIBITRICombine[randIndex == 0,]


################################################# Random Forest #############################################

#UniGram
train_control <- trainControl(method="repeatedcv",number=10, repeats=3,savePrediction=TRUE)

modelRF <- train(PosNeg~., data=trainingData,trControl=train_control, method="rf")
predictionRF<- predict(modelRF,newdata = testingData)

trellis.par.set(caretTheme())
plot(modelRF, metric = "Kappa",main = "Random Forest Plot unigram")

#Bigram

modelRFBigram <- train(PosNeg~., data=trainingDataBigram,trControl=train_control, method="rf")
predictionRFBigram <- predict(modelRFBigram,newdata = testingDataBigram)

trellis.par.set(caretTheme())
plot(modelRFBigram, metric = "Kappa",main = "Random Forest Plot Bigram")

#Trigram 
modelRFTrigram <- train(PosNeg~., data=trainingDataTrigram,trControl=train_control, method="rf")
predictionRFTrigram <- predict(modelRFTrigram,newdata = testingDataTrigram)

trellis.par.set(caretTheme())
plot(modelRFTrigram, metric = "Kappa",main = "Random Forest Plot Trigram")

#Unigram, Bigram and TriGram

modelRFUB <- train(PosNeg~., data=trainingDataUniBi,trControl=train_control, method="rf")
predictionRFUB <- predict(modelRFUB,newdata = testingDataUniBi)

trellis.par.set(caretTheme())
plot(modelRFUB, metric = "Kappa",main = "Random Forest Plot unigram ,bigram and trigram")




############################## Naive Baise #######################################

#UniGram
train_control <- trainControl(method="repeatedcv",number=10, repeats=3,savePrediction=TRUE)

modelNB <- train(PosNeg~., data=trainingData, method="nb")
predictionNB <- predict(modelNB,newdata = testingData)

trellis.par.set(caretTheme())
plot(modelNB, metric = "Kappa",main = "Naiive bayers Plot unigram")

#Bigram

modelNBBigram <- train(PosNeg~., data=trainingDataBigram, method="nb")
predictionNBBigram <- predict(modelNBBigram,newdata = testingDataBigram)

trellis.par.set(caretTheme())
plot(modelNBBigram, metric = "Kappa",main = "Naiive bayers Plot Bigram")

#Trigram 
modelNBTrigram <- train(PosNeg~., data=trainingDataTrigram, method="nb")
predictionNBTrigram <- predict(modelNBTrigram,newdata = testingDataTrigram)

trellis.par.set(caretTheme())
plot(modelNBTrigram, metric = "Kappa",main = "Naiive bayers Plot Trigram")

#Unigram, Bigram and TriGram

modelNBUB <- train(PosNeg~., data=trainingDataUniBi, method="nb")
predictionNBUB <- predict(modelNBUB,newdata = testingDataUniBi)

trellis.par.set(caretTheme())
plot(modelNBUB, metric = "Kappa",main = "Naiive bayers Plot unigram, bigram, trigram")

################################################# KNN #############################################

#UniGram
train_control <- trainControl(method="repeatedcv",number=10, repeats=3,savePrediction=TRUE)

modelKNN <- train(PosNeg~., data=trainingData,trControl=train_control, method="knn")
predictionKNN<- predict(modelKNN,newdata = testingData)

trellis.par.set(caretTheme())
plot(modelKNN, metric = "Kappa",main = "KNN Plot unigram")

#Bigram

modelKNNBigram <- train(PosNeg~., data=trainingDataBigram,trControl=train_control, method="knn")
predictionKNNBigram <- predict(modelKNNBigram,newdata = testingDataBigram)

trellis.par.set(caretTheme())
plot(modelKNNBigram, metric = "Kappa",main = "KNN Plot Bigram")

#Trigram 
modelKNNTrigram <- train(PosNeg~., data=trainingDataTrigram,trControl=train_control, method="knn")
predictionKNNTrigram <- predict(modelKNNTrigram,newdata = testingDataTrigram)

trellis.par.set(caretTheme())
plot(modelKNNTrigram, metric = "Kappa",main = "KNN Plot Trigram")

#Unigram, Bigram and TriGram

modelKNNUB <- train(PosNeg~., data=trainingDataUniBi,trControl=train_control, method="knn")
predictionKNNUB <- predict(modelKNNUB,newdata = testingDataUniBi)

trellis.par.set(caretTheme())
plot(modelKNNUB, metric = "Kappa",main = "KNN Plot unigram ,bigram and trigram")



############################   SVM Class ##################################

#UniGram
train_control <- trainControl(method="repeatedcv",number=10, repeats=3,savePrediction=TRUE)

modelSVM <- train(PosNeg~., data=trainingData,trControl=train_control, method="svmRadial")
predictionSVM <- predict(modelSVM,newdata = testingData)

trellis.par.set(caretTheme())
plot(modelSVM, metric = "Kappa",main = "SVM Plot unigram")

#Bigram

modelSVMBigram <- train(PosNeg~., data=trainingDataBigram,trControl=train_control, method="svmRadial")
predictionSVMBigram <- predict(modelSVMBigram,newdata = testingDataBigram)

trellis.par.set(caretTheme())
plot(modelSVMBigram, metric = "Kappa",main = "SVM Plot Bigram")

#Trigram 
modelSVMTrigram <- train(PosNeg~., data=trainingDataTrigram,trControl=train_control, method="svmRadial")
predictionSVMTrigram <- predict(modelSVMTrigram,newdata = testingDataTrigram)

trellis.par.set(caretTheme())
plot(modelSVMTrigram, metric = "Kappa",main = "SVM Plot Trigram")

#Unigram, Bigram and TriGram

modelSVMUB <- train(PosNeg~., data=trainingDataUniBi,trControl=train_control, method="svmRadial")
predictionSVMUB <- predict(modelSVMUB,newdata = testingDataUniBi)

trellis.par.set(caretTheme())
plot(modelSVMUB, metric = "Kappa",main = "SVM Plot unigram ,bigram and trigram")

############################ Confusion Matrix for All 4 Algorithm ###################

confusionMatrix(predictionRF,testingData$PosNeg)
confusionMatrix(predictionNB,testingData$PosNeg)
confusionMatrix(predictionKNN,testingData$PosNeg)
confusionMatrix(predictionSVM,testingData$PosNeg)


confusionMatrix(predictionRFBigram,testingDataBigram$PosNeg)
confusionMatrix(predictionNBBigram,testingDataBigram$PosNeg)
confusionMatrix(predictionKNNBigram,testingDataBigram$PosNeg)
confusionMatrix(predictionSVMBigram,testingDataBigram$PosNeg)

confusionMatrix(predictionRFTrigram,testingDataTrigram$PosNeg)
confusionMatrix(predictionNBTrigram,testingDataTrigram$PosNeg)
confusionMatrix(predictionKNNTrigram,testingDataTrigram$PosNeg)
confusionMatrix(predictionSVMTrigram,testingDataTrigram$PosNeg)

confusionMatrix(predictionRFUB,testingDataUniBi$PosNeg)
confusionMatrix(predictionNBUB,testingDataUniBi$PosNeg)
confusionMatrix(predictionKNNUB,testingDataUniBi$PosNeg)
confusionMatrix(predictionSVMUB,testingDataUniBi$PosNeg)

######################## K-fold #########################

mean(genarrayperf(modelRF$pred))
mean(genarrayperf(modelRFBigram$pred))
mean(genarrayperf(modelRFTrigram$pred))
mean(genarrayperf(modelRFUB$pred))
mean(genarrayperf(modelNB$pred))
mean(genarrayperf(modelNBBigram$pred))
mean(genarrayperf(modelNBTrigram$pred))
mean(genarrayperf(modelNBUB$pred))
mean(genarrayperf(modelKNN$pred))
mean(genarrayperf(modelKNNBigram$pred))
mean(genarrayperf(modelKNNTrigram$pred))
mean(genarrayperf(modelKNNUB$pred))
mean(genarrayperf(modelSVM$pred))
mean(genarrayperf(modelSVMBigram$pred))
mean(genarrayperf(modelSVMTrigram$pred))
mean(genarrayperf(modelSVMUB$pred))







