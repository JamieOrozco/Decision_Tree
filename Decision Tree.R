# Use the Titanic dataset to determine who would survive

# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("randomForest")

require(caret)
require(rpart)
require(rpart.plot)
require(randomForest)

titanic.df <-read.csv("titanic.csv", fileEncoding = "UTF-8-BOM")

titanic.df$Pclass <- factor(titanic.df$Pclass, levels = c(1, 2, 3), 
                            labels = c("Upper", "Middle", "Lower"))
titanic.df$Survived <- factor(titanic.df$Survived, levels = c(0, 1), 
                          labels = c("Perished", "Survived"))

colSums(is.na(titanic.df))

#below is used because the above code says missing data
titanic.df$Age <- ifelse(is.na(titanic.df$Age), 
                mean(titanic.df$Age, na.rm=TRUE), titanic.df$Age)

set.seed(200)
train.index <- sample(c(1:dim(titanic.df)[1]), dim(titanic.df)[1]*0.7)  

# Columns that do not seem to provide signal
# such as passenger ID
unwanted.columns <- c(1, 4, 9, 11, 12) 

train.df <- titanic.df[train.index, -unwanted.columns]
valid.df <- titanic.df[-train.index, -unwanted.columns]

# Verify similar distribution

prop.table(table(train.df$Survived))
prop.table(table(valid.df$Survived))

# Create the tree and plot
tree <- rpart(Survived ~ ., data = train.df, method = 'class')

rpart.plot(tree, extra = 106)
prp(tree, faclen = 0, cex = 0.8, extra = 1)

tot_count <- function(x, labs, digits, varlen)
{
  paste(labs, "\n\nn =", x$frame$n)
}
prp(tree, faclen = 0, cex = 0.8, node.fun=tot_count)

only_count <- function(x, labs, digits, varlen)
{
  paste(x$frame$n)
}

boxcols <- c("pink", "palegreen3")[tree$frame$yval]

par(xpd=TRUE)
prp(tree, faclen = 0, cex = 0.8, node.fun=only_count, box.col = boxcols)
legend("bottomleft", legend = c("died","survived"), 
       fill = c("pink", "palegreen3"), title = "Group")

pred.values <-predict(tree, valid.df, type = 'class')

confusionMatrix(as.factor(pred.values), as.factor(valid.df$Survived), 
                positive = "Survived")

# Random forest
rf <- randomForest(Survived ~ ., data = train.df, ntree = 500, 
                   mtry = 5, nodesize = 5, importance = TRUE) 

# Adjust different levels of mtry hyper-parameter

# variable importance plot
varImpPlot(rf, type = 1)

rf.pred <- predict(rf, valid.df)

confusionMatrix(as.factor(rf.pred), as.factor(valid.df$Survived), 
                positive = "Survived")




