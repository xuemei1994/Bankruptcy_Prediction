rm(list = ls()) # delete objects 
cat("\014") # clear console library(tidyverse)

#install.packages("cowplot")
library(cowplot)
library(gridExtra)
library(pROC)
library(glmnet) 
library(latex2exp)
library(randomForest)
library(ROCR)
library(tidyverse)

df = read.csv('data.csv', header=1) 
df= within(df,rm('Net.Income.Flag','Liability.Assets.Flag')) # binary variables , all 1 and 0 
dim(df) # 6819 x 96, total 6819 observations
str(df)
summary(df)

set.seed(1)
M = 3
n = dim(df)[1]   # 6819
p = dim(df)[2]-1 # 95

X = as.matrix(df[,-1])
y = as.matrix(df[, 1])

train_auc = matrix(NA,M,4) # order: lasso, en, ridge, rf
test_auc  = matrix(NA,M,4) # order: lasso, en, ridge, rf

time.eln  = matrix(NA,M,1)
time.las  = matrix(NA,M,1)
time.rid  = matrix(NA,M,1)


for (m in 1:M) {
  
    ###################################################################################
    ###################################################################################
  
    ### Selecting Training and Testing Sets

    all.P.index         =    which(df$Bankrupt. == 1)        # 220: all positive index
    shuffled.P.indexes  =    sample(all.P.index)             # shuffle
    train.P.index       =    shuffled.P.indexes[1:198]       # 198: first 0.9*n+ for train
    test.P.index        =    shuffled.P.indexes[199:220]     # 22:  rest for test 

    all.N.index         =    which(df$Bankrupt. == 0)        # 6599: all negative index
    shuffled.N.indexes  =    sample(all.N.index)             # shuffle
    train.N.index       =    shuffled.N.indexes[1:5939]      # 5939: first 0.9*n- for train
    test.N.index        =    shuffled.N.indexes[5940:6599]   # 660:  rest for test

    train               =    c(train.P.index, train.N.index) # 6137 = 198 + 5939
    test                =    c(test.P.index, test.N.index)   # 682  = 22 + 660 
    
    X.train             =    X[train, ]
    y.train             =    y[train]
    X.test              =    X[test, ]
    y.test              =    y[test]
    
    # weights
    w                   =    ifelse(y.train==1, 1, 198/5939)

    ###################################################################################
    ###################################################################################
    
    ### Fitting Models
    
    # -----lasso model----- #
    time0.las   =     proc.time()
    cv.lasso    =     cv.glmnet(X.train, y.train, family = "binomial", alpha = 1,  nfolds = 10, type.measure="auc", weights=w)
    time1.las   =     proc.time()
    lasso       =     glmnet(X.train, y.train, lambda = cv.lasso$lambda.min, family = "binomial", alpha = 1, weights=w)
    time.las[m,1] = (time1.las - time0.las)[['elapsed']]
    
    fit = lasso
    beta0.hat               =        fit$a0
    beta.hat                =        as.vector(fit$beta)
    prob.train = exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
    prob.test  = exp(X.test  %*% beta.hat +  beta0.hat  )/(1 + exp(X.test  %*% beta.hat +  beta0.hat  ))
    
    # calculate auc
    df_new1 = data.frame(y.train, prob.train)
    df_new2 = data.frame(y.test, prob.test)
    r1 = roc(df_new1$y.train, df_new1$prob.train, levels = c(0, 1), direction = '<')
    r2 = roc(df_new2$y.test, df_new2$prob.test, levels = c(0, 1), direction = '<' )
    
    # recording auc
    train_auc[m,1] = round(r1$auc,5)
    test_auc[m,1]  = round(r2$auc,5)
    cat('Lasso:', m, 'train auc:', round(r1$auc,5), 'test auc:', round(r2$auc,5), 'df:', fit$df, '\n')
    
    #plotting cv curve
    par(mfrow = c(1,3)) # arrange plots into 1x2 format
    plot.las  = plot(cv.lasso, main='lasso')
    

    
    # -----elastic-net model-----#
    time0.eln   =     proc.time()
    cv.elnet    =     cv.glmnet(X.train, y.train, family = "binomial", alpha = 0.5,  nfolds = 10, type.measure="auc", weights=w)
    time1.eln   =     proc.time()
    elnet       =     glmnet(X.train, y.train, lambda = cv.elnet$lambda.min, family = "binomial", alpha = 0.5, weights=w)
    time.eln[m,1] = (time1.eln - time0.eln)[['elapsed']]
    
    fit = elnet
    beta0.hat               =        fit$a0
    beta.hat                =        as.vector(fit$beta)
    prob.train = exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
    prob.test  = exp(X.test  %*% beta.hat +  beta0.hat  )/(1 + exp(X.test  %*% beta.hat +  beta0.hat  ))
    
    # calculate auc
    df_new1 = data.frame(y.train, prob.train)
    df_new2 = data.frame(y.test, prob.test)
    r1 = roc(df_new1$y.train, df_new1$prob.train, levels = c(0, 1), direction = '<')
    r2 = roc(df_new2$y.test, df_new2$prob.test, levels = c(0, 1), direction = '<' )
    
    # recording auc
    train_auc[m,2] = round(r1$auc,5)
    test_auc[m,2]  = round(r2$auc,5)
    cat('El-Net:', m, 'train auc:', round(r1$auc,5), 'test auc:', round(r2$auc,5), 'df:', fit$df, '\n')
    
    #plotting cv curve
    plot.eln = plot(cv.elnet, main='elastic-net') 
    
    
    
    # -----ridge model----- #
    time0.rid   =     proc.time()
    cv.ridge    =     cv.glmnet(X.train, y.train, family = "binomial", alpha = 0,  nfolds = 10, type.measure="auc", weights=w)
    time1.rid   =     proc.time()
    ridge       =     glmnet(X.train, y.train, lambda = cv.ridge$lambda.min, family = "binomial", alpha = 0, weights=w)
    time.rid[m,1] = (time1.rid - time0.rid)[['elapsed']]

    fit = ridge
    beta0.hat               =        fit$a0
    beta.hat                =        as.vector(fit$beta)
    prob.train = exp(X.train %*% beta.hat +  beta0.hat  )/(1 + exp(X.train %*% beta.hat +  beta0.hat  ))
    prob.test  = exp(X.test  %*% beta.hat +  beta0.hat  )/(1 + exp(X.test  %*% beta.hat +  beta0.hat  ))
    
    # calculate auc
    df_new1 = data.frame(y.train, prob.train)
    df_new2 = data.frame(y.test, prob.test)
    r1 = roc(df_new1$y.train, df_new1$prob.train, levels = c(0, 1), direction = '<')
    r2 = roc(df_new2$y.test, df_new2$prob.test, levels = c(0, 1), direction = '<' )
    
    # recording auc
    train_auc[m,3] = round(r1$auc,5)
    test_auc[m,3]  = round(r2$auc,5)
    cat('Ridge:', m, 'train auc:', round(r1$auc,5), 'test auc:', round(r2$auc,5), 'df:', fit$df, '\n')
    
    #plotting cv curve
    plot.rid = plot(cv.ridge, main='ridge') 
    
    
    
    # -----random forest model----- #
    time0.rf   =     proc.time()
    rf = randomForest(x=X.train, y=as.factor(y.train), mtry=floor(sqrt(p)),ntree=100,nodesize=25, importance=F) 
    time1.rf   =     proc.time()
    prob.train = as.numeric(as.character(predict(rf,
                                                  X.train,type="response")))
    prob.test = as.numeric(as.character(predict(rf,
                                                X.test,type="response")))
    
    rf.prob.train = prediction(prob.train,y.train)
    rf.prob.test = prediction(prob.test,y.test)
    
    r1 <- performance(rf.prob.train, measure = "auc")@y.values[[1]]
    r2 <- performance(rf.prob.test, measure = "auc")@y.values[[1]]
    
    
    # recording auc
    train_auc[m,4] = round(r1,5)
    test_auc[m,4]  = round(r2,5)
    cat('RF:', m, 'train auc:', round(r1,5), 'test auc:', round(r2,5))
    
    
}  



# ----- PART3(b) -----
par(mfrow = c(1,2)) # arrange plots into 1x2 format

colnames(train_auc) = c('Lasso', 'EN', 'Ridge', 'RF')
colnames(test_auc)  = c('Lasso', 'EN', 'Ridge', 'RF')
lowest_auc = min(train_auc, test_auc)
boxplot(train_auc, main='Train AUC', ylim=c(0.5,1))
boxplot(test_auc, main='Test AUC', ylim=c(0.5,1))


# ----- PART3(c) -----
# single model running time recording - 50 samples average
ave.time.eln   = mean(time.eln)   # one elastic-net running time
ave.time.las   = mean(time.las)   # one lasso running time
ave.time.rid   = mean(time.rid)   # one ridge model running time



########################################################################################
########################################################################################
########################################################################################
####################################### ALL DATA #######################################
########################################################################################
########################################################################################
########################################################################################

# weights
w                   =    ifelse(y==1, 1, 220/6599)

############     LASSO - All DATA ############
set.seed(1)
timeLas0    =      proc.time()
cv.lasso    =      cv.glmnet(X, y, family = "binomial", alpha = 1,  nfolds = 10, type.measure="auc", weights=w)
lasso.fit   =      glmnet(X, as.vector(y), lambda = cv.lasso$lambda.min, family = "binomial", alpha = 1, weights=as.vector(w))
timeLas1    =      proc.time()

#######coefficient. lasso 
df.lasso    =      cv.lasso$nzero[which.min(cv.lasso$cvm)]
a0.hat.lasso      =   lasso.fit$a0[lasso.fit$lambda==cv.lasso$lambda.min]
beta.hat.lasso    =   lasso.fit$beta[ ,lasso.fit$lambda==cv.lasso$lambda.min]

############     RIDGE - All DATA ############
set.seed(1)
timeRid0    =      proc.time()
cv.ridge    =      cv.glmnet(X, y, family = "binomial", alpha=0, nfolds = 10, type.measure="auc", weights=w)
ridge.fit   =      glmnet(X, as.vector(y), lambda = cv.ridge$lambda.min, family = "binomial", alpha = 0, weights=as.vector(w))
timeRid1    =      proc.time()

#######coefficient. ridge
df.ridge    =      cv.ridge$nzero[which.min(cv.ridge$cvm)]
a0.hat.ridge      =   ridge.fit$a0[ridge.fit$lambda==cv.ridge$lambda.min]
beta.hat.ridge    =   ridge.fit$beta[ ,ridge.fit$lambda==cv.ridge$lambda.min]

############     ELNET - All DATA ############
set.seed(1)
timeEln0    =      proc.time()
cv.elnet    =      cv.glmnet(X, y, family = "binomial", alpha=0.5, nfolds = 10, type.measure="auc", weights=w)
elnet.fit   =      glmnet(X, as.vector(y), lambda = cv.elnet$lambda.min, family = "binomial", alpha = 0.5, weights=as.vector(w))
timeEln1    =      proc.time()

#######coefficient. elnet
df.elnet    =      cv.elnet$nzero[which.min(cv.elnet$cvm)]
a0.hat.elnet      =   elnet.fit$a0[elnet.fit$lambda==cv.elnet$lambda.min]
beta.hat.elnet    =   elnet.fit$beta[ ,elnet.fit$lambda==cv.elnet$lambda.min]



############     RANDOM FOREST - ALL data #########
set.seed(1)
timeRf0 = proc.time()
rf.full  =  randomForest(Bankrupt.~., data=df, mtry=floor(sqrt(p)), importance=T)
timeRf1 = proc.time()






betaS.en               =     data.frame(colnames(X), as.vector(beta.hat.elnet))
colnames(betaS.en)     =     c( "feature", "Coefficient")

betaS.ls               =     data.frame(colnames(X), as.vector(beta.hat.lasso))
colnames(betaS.ls)     =     c( "feature", "Coefficient")

betaS.rg               =     data.frame(colnames(X), as.vector(beta.hat.ridge))
colnames(betaS.rg)     =     c( "feature", "Coefficient")

#betaS.rf               =     data.frame(colnames(df)[1:p], as.vector(rf.full$importance[,1]))
betaS.rf               =     data.frame(colnames(X), as.vector(importance(rf.full)[,1]))
colnames(betaS.rf)     =     c( "feature", "PctIncMSE")



betaS.ls$feature     =  factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$Coefficient, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$Coefficient, decreasing = TRUE)])
betaS.rg$feature     =  factor(betaS.rg$feature, levels = betaS.en$feature[order(betaS.en$Coefficient, decreasing = TRUE)])
betaS.rf$feature     =  factor(betaS.rf$feature, levels = betaS.en$feature[order(betaS.en$Coefficient, decreasing = TRUE)])


L = min(betaS.ls$Coefficient, betaS.en$Coefficient, betaS.rg$Coefficient)
U = max(betaS.ls$Coefficient, betaS.en$Coefficient, betaS.rg$Coefficient)
U_rf = max(betaS.rf$PctIncMSE)
L_rf = min(betaS.rf$PctIncMSE)

lsPlot =  ggplot(betaS.ls, aes(x=feature, y=Coefficient)) +
  geom_bar(stat = "identity", fill=ifelse(betaS.ls$Coefficient>0,"steelblue","red"), colour="black") +
  ggtitle(("Lasso")) + ylim(c(L,U))+theme(axis.text.x = element_text(size = 0,angle=90),
                                          axis.title.x = element_blank())


enPlot =  ggplot(betaS.en, aes(x=feature, y=Coefficient)) +
  geom_bar(stat = "identity", fill=ifelse(betaS.en$Coefficient>0,"steelblue","red"), colour="black")  + 
  ggtitle("Elastic-Net")+ ylim(c(L,U))+ theme(axis.text.x = element_text(size = 0,angle=90),
                                              axis.title.x = element_blank())


rgPlot =  ggplot(betaS.rg, aes(x=feature, y=Coefficient)) +
  geom_bar(stat = "identity", fill=ifelse(betaS.rg$Coefficient>0,"steelblue","red"), colour="black") + 
  ggtitle("Ridge")+ ylim(c(L,U)) + theme(axis.text.x = element_text(size = rel(0),angle=90),
                                         axis.title.x = element_blank())  


rfPlot =  ggplot(betaS.rf, aes(x=feature, y=PctIncMSE)) +
  geom_bar(stat = "identity", fill=ifelse(betaS.rf$PctIncMSE>8,"green","steelblue"), colour="black")+ 
  ggtitle("Random Forest")+ ylim(c(L_rf,U_rf))+ theme(axis.text.x = element_text(size =rel(0.95),angle=90))

plot_grid(enPlot, lsPlot, rgPlot, rfPlot,align = "v", nrow = 4, rel_heights = c(1/6, 1/6, 1/6, 1/2))

######################################
timeLas = timeLas1 - timeLas0
timeRid = timeRid1 - timeRid0
timeEln = timeEln1 - timeEln0
timeRf  = timeRf1 - timeRid0

answer_table = matrix(NA,4,2)
colnames(answer_table) = c("Median Test AUC","Time to Fit")
rownames(answer_table) = c("Elastic-Net","Lasso","Ridge","Random Forest")
answer_table[1,2] = round(timeEln["elapsed"],2)
answer_table[2,2] = round(timeLas["elapsed"],2)
answer_table[3,2] = round(timeRid["elapsed"],2)
answer_table[4,2] = round(timeRf["elapsed"],2)


test_auc = as.data.frame(test_auc)

answer_table[1,1] = median(train_auc[,2])
answer_table[2,1] = median(train_auc[,1])
answer_table[3,1] = median(train_auc[,3])
answer_table[4,1] = median(train_auc[,4])

answer_table
