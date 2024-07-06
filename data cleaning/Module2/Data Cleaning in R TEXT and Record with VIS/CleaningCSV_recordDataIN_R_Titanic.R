## Gates, 2020 #####################################
##     DATA CLEANING
##     1) Reading in csv data
##     2) Looking at the data frame
##     3) Libraries
##     4) Setting a WD
##     5) installing
##     6) check for missing values
##     7) visual EDA part 1 (ggplot2)
##     8) Look at data types (str)
##     9) Normalization
##     10) Discretization (binning)
##     11) Feature Generation
##    
## The DATA
##   https://drive.google.com/file/d/1gwaRiogV2wQiJchLVDMLqxeVn_-jeBYW/view?usp=sharing
#####################################################
## DO THIS ONCE for any package that you do not have.
## install.packages("ggplot2")
library(ggplot2)
library(tidyverse)

## Set your working director to the path were your code AND datafile is
setwd("D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Module2/Data Cleaning in R TEXT and Record with VIS")


## Read in .csv data
## Reference:
## https://stat.ethz.ch/R-manual/R-devel/library/utils/html/read.table.html

filename="Titanic_Training_Data.csv"
MyTitanicData <- read.csv(filename, header = TRUE, na.strings = "NA")

#######################################################
## Clean in smart steps
##
## For example, a good first step is to look at all the columns/variables
## one at a time. Do you plan to remove any?
## If yes, remove them so that you do not need to further clean 
## or worry about them.
## 
## CONSIDER
##
## 1) Data types
## 2) Do numbers of levels look right
## 3) Missing values
## 4) Incorrect values or poor formats
## 5) Outliers
## 6) Feature generation
## 7) Normalization, etc..
##
## Note - this methods works well with up to 25 columns - so not for highD or
## text data, which is cleaned in other ways.
## Also, while record data can be cleaned this way, other formats, such as
## transaction data or sequential data cannot. 
## Remember - no part of Data Science is a "black box". YOU are always
## the brains.
#############################################################################

## ------------- ##
## Open the dataset and look at it. Yes, data can be huge, but you can and
## should always have a look. 

## Look at the data as a data frame
## Look at the data types
head(MyTitanicData)
str(MyTitanicData)

## The str of the data (the types) can answer many questions.
## What is wrong here?
## 1) Pclass should be factor, not int
## 2) Name has no use and should be removed.
## 3) Sex should not have 3 levels.
## 4) Age in type num, which is good. We are lucky and
##    can see the -7 which is not correct. Do not depend
##    on this as data can have 100,000s of rows.
## 5) Ticket is not a useful variable - we will remove it.
## 6) Fare shows as "Factor" but should be num. This means
##    there are errors in it. 
## 7) Cabin and Embarked and PassengerID
##    have no use and should be removed.
###########################################################
## OK - let's fix all of the above first and then continue...
################################################################

## Print the variables:
colnames(MyTitanicData)

## Remove all columns/variables not needed.
head(MyTitanicData)
MyTitanicData <-MyTitanicData[-c(1,4,9,11,12)]
## Look again...
head(MyTitanicData)

## Good - now we have a nice dataframe to clean up. 

(MyColNames<-names(MyTitanicData))
#######################################
## Fix data types   ###################
#######################################
## Look at the type of each
str(MyTitanicData)
## Sometimes it also helps to look at all the variables in
## a table.
## Tables are great!
## loops - make all the tables at once
for(i in 1:ncol(MyTitanicData)){
  print(table(MyTitanicData[i]))
}
##
## From the tables, we see that for Survived, we have 
## a word, "banana", which shoud not be there.
## We cannot fix this because Survived is our LABEL.
## Let's remove the row with banana
nrow(MyTitanicData)
MyTitanicData<-MyTitanicData[-which(MyTitanicData$Survived=="banana"),  ]
nrow(MyTitanicData)
str(MyTitanicData$Survived)

## Drop unused levels #####################
MyTitanicData$Survived<-factor(MyTitanicData$Survived)
head(MyTitanicData$Survived)
table(MyTitanicData$Survived)

#######  OK --------------
## Next, always check back to see where you are:

head(MyTitanicData)
str(MyTitanicData)

## The variable (LABEL) "Survived" looks fine.
## Check it for missing values now and
## for any incorrect values:
table(MyTitanicData$Survived)
(sum(is.na(MyTitanicData$Survived)))

## Perfect - no missing values and no incorrect
## or odd values.

#######  Next variable to look at is Pclass #######
#################################################

str(MyTitanicData$Pclass)
## Notice that right now, Pclass is an int
## This is incorrect. Pclass should be
## type factor. 
## Make this change:
MyTitanicData$Pclass<-as.factor(MyTitanicData$Pclass)
## Check it
table(MyTitanicData$Pclass)
str(MyTitanicData$Pclass)

## OK - good. Now check it for missing values
(sum(is.na(MyTitanicData$Pclass)))

############# Next variable is Sex ###########
str(MyTitanicData$Sex)
## Here, we have a problem. Sex should have two levels
## "male" and "female".
## However, the str shows 3 levels with one as "M".
## Because we can be safe with assuming that M is male
## we can correct this and update the levels:

MyTitanicData$Sex[MyTitanicData$Sex=="M"]<-"male"
MyTitanicData$Sex
## Drop the empty level
MyTitanicData$Sex<-factor(MyTitanicData$Sex)
head(MyTitanicData$Sex)
table(MyTitanicData$Sex)
## Check for missing values
(sum(is.na(MyTitanicData$Sex)))


########  Next variable is Age ###################
str(MyTitanicData$Age)
## Good and bad. Age is type num. That is good. 
## However, we need to make sure that all ages
## are in a correct range: 0 - 120.
## Let's do this and if any are not, convert to NA
## Count the NAs first
(sum(is.na(MyTitanicData$Age)))
MyTitanicData$Age[MyTitanicData$Age>120 | MyTitanicData$Age<0]<-NA
## Count the NAs again and use a table to check the ages
(sum(is.na(MyTitanicData$Age)))
table(MyTitanicData$Age)
## OK - what happened?
## First, we corrected one age and made it NA.
## Next, using table, we see that some ages are decimals.
## My researching this, we learn that Titanic data ages
## can be decimals and values >0 and <1 are babies.
## However, this is a good time to first remove 
## missing values and then create a new binned
## variable for age groups. This is called discretization
## and feature generation. 

## Missing values for AGE. 
## For Titanic, the age matters. So, replacing missing
## ages with the mean or median is not a good idea.

## What percentage are missing?
## total rows
(AllRows<-nrow(MyTitanicData))
(MissingAgeRows<-sum(is.na(MyTitanicData$Age)))
(PercentMissing<-MissingAgeRows/AllRows)

## Hmmm - so, removing all rows with a missing
## age will create a loss of 20% of the data. That is
## a lot. There is not perfect answer to issues like this.
## If we wanted to remove all rows for which the age is
## missing, here is how we would do it:
## BEFORE
nrow(MyTitanicData)
MyTitanicData<-MyTitanicData[-which(is.na(MyTitanicData$Age)),]
## AFTER
nrow(MyTitanicData)

########## NOTE: Code will also be shown below ##########
##               for how to replace values with
##               a mean or median, etc....             ##
#########################################################

## Now - let's look at Age to assure that it is good.
str(MyTitanicData$Age)
table(MyTitanicData$Age)
sum(is.na(MyTitanicData$Age))
## OK - we are all set.

################################################
## Discretization- Binning - Feature Generation 
###########################################################
## To learn more about how AGE is affecting survival, it is 
## interesting to create a new variable called AGEGroup
MyTitanicData$AgeGroup <- 
  cut(MyTitanicData$Age, breaks=c(-Inf, 9, 15, 25, 40, Inf), 
      labels=c("Under9","NineTo15", "FifteenTO25","TwentyFiveTO40", "Over40"))

head(MyTitanicData)

#########  The next variables are Sibsp and Parch ###
## Using Google we can learn that:
## Sibsp Number of Siblings/Spouses Aboard
## Parch Number of Parents/Children Aboard
#####################################################
## So, let's AGGREGATE!
## Let's create a new vairable (this is called feature generation)
## that is the SUM of Sibsp and Parch.

## Why?? is this always right to do??
## NO!
## Nothing is "always right to do". We are doing this to learn
## how to do it. The when, if, why, and whether is always situation
## dependent.

MyTitanicData$Family<-MyTitanicData$SibSp+MyTitanicData$Parch
(MyTitanicData)
str(MyTitanicData)

## Check for missing values for Sibsp, Parch, and our
## new variable called Family. Also check for odd values:

table(MyTitanicData$SibSp)
## Sibsp looks fine. However, it is NOT BALANCED
## Notice that most data is 0 or 1. 
## In some cases, it might be interesting to remove
## this column.
table(MyTitanicData$Parch)  ## See note above.
table(MyTitanicData$Family)
## While we will not do this here, it is interesting
## to consider the creation of a new variable
## with three levels: no fam, small fam, large fam
## or similar. Think about why and when this might 
## be better.

sum(is.na(MyTitanicData$SibSp))
sum(is.na(MyTitanicData$Parch))
sum(is.na(MyTitanicData$Family))

### OK - all set! ################

################################################
## Now - where are we?  - Let's look:
###############################################
str(MyTitanicData)
sum(is.na(MyTitanicData))

###########  NExt and last variable:  Fare #########
####################################################
str(MyTitanicData$Fare)
sum(is.na(MyTitanicData$Fare))
table(MyTitanicData$Fare)

## Interesting! First, the data type is wrong.
## The type shows as Factor. However, we know
## that Fare is a dollar value and so should be 
## num.
## This also gives us a hint that it is likely
## that inside the data for Fare is a non-number.
## Next, a Fare is a cost. We should assure that
## all Fare values are >0 and perhaps < some value.
## We can also check for outliers.

## So, let's start with finding non-numbers and replacing
## them with NA.

## We may also want to round values to the nearest
## whole number - or even group/bin them...

## Remove non-numbers and change type to num:
## BEFORE
str(MyTitanicData$Fare)
MyTitanicData$Fare
## Notice the need for as.character first ##
MyTitanicData$Fare<-round(as.numeric(as.character(MyTitanicData$Fare)),0)
## A warning will appear that notes the addition of NA
## This is OK.
str(MyTitanicData$Fare)
MyTitanicData$Fare

## This is a great spot to use boxplots to look for
## outliers.
boxplot(MyTitanicData$Fare)
## Let's also get the mean, median, max, and min
summary(MyTitanicData$Fare)
## This suggests that values above 500 are very likely to be outliers
## An interesting question is what the Pclass is for
## the row with a Fare of 512.
## Let's find out....
MyTitanicData[MyTitanicData$Fare>400,]
## Hmmm - there are three of them and they are all
## first class. This is NOT an outlier!! We cannot remove it.

## OK - now, let's check the missing values in Fare
sum(is.na(MyTitanicData$Fare))
## There are 2. We must correct them. 
## This is a good chance to replace them with
## the mean or median of the correct class-fare.
## In other words, if a value for Fare is missing
## we should first check the Pclass. Then, replace
## the missing value with the mean or median for
## that Pclass....

## HOW TO see rows for which a specific variable value is missing - such as Fare
(MissingFare_DF <- MyTitanicData[is.na(MyTitanicData$Fare),])


################################
### GUT CHECK ###
## Missing values in Fare....
(sum(is.na(MyTitanicData$Fare)))
## The variance before we fix the 
## missing Fare values....and we will
## get the var AFTER as well.
(FareVar1<-var(MyTitanicData$Fare, na.rm=TRUE))
####################################

########  Recall - here we will first replace the
##        missing (NA) Fare value for the row with
##        Pclass=1. We will then repeat this for 
##        the row with Pclass=3. 
#######
## Fare=NA ROWs for which Pclass is 1
(PclassDF_Temp<-MyTitanicData[MyTitanicData$Pclass==1,])
## The average Fare when Pclass is 1
(FareMean<-mean(PclassDF_Temp$Fare, na.rm=TRUE))
## NOW - replace the missing Fare with this FareMean
MyTitanicData$Fare[is.na(MyTitanicData$Fare) 
                   & MyTitanicData$Pclass=="1"] <- FareMean
## Check to assure the missing value in Fare was updated...
(sum(is.na(MyTitanicData$Fare)))

## OK - that was for the missing Fare with Pclass=1
## Now we do the same thing for the missing fare
## with Pclass=3. 

(PclassDF_Temp3<-MyTitanicData[MyTitanicData$Pclass==3,])
## The average Fare when Pclass is 1
(FareMean3<-mean(PclassDF_Temp3$Fare, na.rm=TRUE))
## NOW - replace the missing Fare with this FareMean
MyTitanicData$Fare[is.na(MyTitanicData$Fare) 
                   & MyTitanicData$Pclass=="3"] <- FareMean3
## Check to assure the missing value in Fare was updated...
(sum(is.na(MyTitanicData$Fare)))
## OK!! Now we have no NAs for Fare and we have
## replaced the missing values with smart values.

## Get the variance AFTER you cleaned the Fare
(FareVar2<-var(MyTitanicData$Fare, na.rm=TRUE))
## Compare the before and after variance:
cat("BEFORE VAR", FareVar1, " and AFTER Var ", FareVar2, "\n")
## This result should make sense. My changing missing values
## to means, the variance (avg. dist to mean) will reduce.

## Check the dataset again for missing values..
sum(is.na(MyTitanicData))

################
## Write the clean dataframe to a new csv file
#####################################################
PATH="DATA/CleanTitanicLabeled.csv"
write.csv(MyTitanicData,PATH)

######################################################
## We are clean!! We have 0 missing values
## All data types and levels are correct.
## All odd or incorrect values have been managed.

## As another option....
## library(tidyverse)
## This is a fast a fun way to get all the sums of all the NA for
## all the variables at once.
##(map(MyTitanicData, ~sum(is.na(.))))
#############################################
## Before and After Cleaning Measures
############################################
## This is a topic that can take days
## The short message is to calculate
## measures - such as mean and vairable
## as well as total number of rows and
## overall data balance - each time
## you "clean" something. 
##



###############################################################
## As a faster method - 
## The following replaces all missing numeric
## values with the median. This is NOT always
## the right thing to do. Rather, it is an 
## example of HOW to do this if you wish.
##################################################

# for(varname in names(MyTitanicData)){
#   ## Only check numeric variables
#   if(sapply(MyTitanicData[varname], is.numeric)){
#     cat("\n", varname, " is numeric\n")
#     ## Get median
#     (Themedian <- sapply(MyTitanicData[varname],FUN=median, na.rm=TRUE))
#     print(Themedian)
    
# ## check/replace if the values are <=0 
#     ########################################################
#     ## !!! It is NOT always OK to replace missing values !!!
#     ########################################################
#     MyTitanicData[varname] <- replace(MyTitanicData[varname], MyTitanicData[varname] < 0, Themedian)
#     MyTitanicData[varname] <- replace(MyTitanicData[varname], is.na(MyTitanicData[varname]), Themedian)
#   }
#   
# }
########################################################



################################################
## VISUALIZATION and EDA: Looking at the data
################################################
##Let's look at the Pclass data and Fare data

## Recall the dataset and column names
str(MyTitanicData)
head(MyTitanicData)

## Different vis options are more (or less) appropriate
## for different types of data.
## For example, boxplots, histograms, scatterplots, etc.
## are best for numeric data.
## Alternatively, bar graphs and pies are better for 
## qualitative data.

## Scatter plots use two variables. Boxplots can (but
## do not need) to use two variables.

library(ggplot2)

## Bar Graphs.............
TTBaseGraph <- ggplot(MyTitanicData)
## Basic
(MyG1<-TTBaseGraph + geom_bar(aes(Survived, fill = Sex)) + ggtitle("Survived by Gender"))
## Stacked
(MyG2<-TTBaseGraph + geom_bar(aes(Pclass, fill = Sex)) + ggtitle("Class by Gender"))
## Horizontal and theme

## Grouped
(MyG3<-TTBaseGraph + geom_bar(aes(Survived, fill = AgeGroup), position="dodge")+ggtitle("Survived and Gender"))

library(gridExtra)
grid.arrange(MyG1, MyG2, MyG3, nrow = 2)

######### Radial ..............
(MyR1<-ggplot(data=MyTitanicData,aes(x=Survived,fill=Sex))+
    geom_bar()+ coord_polar()+ ggtitle("Survived and Gender")+
    scale_fill_manual(values=c("orange", "green", "red"))+xlab("")+ylab(""))

(MyR2 <- ggplot(MyTitanicData, aes(x = Age, fill = AgeGroup)) +
    geom_bar(width = 1) +
    coord_polar(theta = "x"))


grid.arrange(MyR1, MyR2, nrow = 2)

########### Violins ....................
medAge4=median(MyTitanicData$Age)

(MyV1 <- ggplot(MyTitanicData, aes(x=Survived, y=Age, fill=Sex)) + 
    geom_violin(trim=TRUE)+ geom_boxplot(width=0.1)+
    geom_text(aes(y = medAge4, label = medAge4), 
              size = 2.5, vjust = 4, hjust=-1)+
    ggtitle("Survival"))

# Rotate the violin plot
(MyV1 + coord_flip())

## facet grid
(MyH1<-ggplot(MyTitanicData,aes(x=Age, fill=AgeGroup))+ geom_density()+
    facet_grid(~AgeGroup)+theme(axis.text.x=element_blank()) +
    ggtitle("Age Groups"))

## JItter
(MyJ1<-ggplot(MyTitanicData,aes(x=Survived, y=Age, col=Sex))+ 
    geom_jitter(stat = "identity",position = "jitter")+
    facet_grid(~Survived)+theme(axis.text.x=element_blank()) +
    ggtitle("Jitter Plots: Survival and Age with Gender"))

## Layer jitter + box
(MyL1<-ggplot(MyTitanicData, aes(x=AgeGroup, y=Fare))+
    geom_boxplot()+
    geom_jitter(position=position_jitter(.01), aes(color=Survived))+
    ggtitle("Fare, Age, and Survival"))


## Bubble
(MyS1<-ggplot(MyTitanicData, aes(x=Age, y=Fare, color=Survived))+
    geom_point(aes(size=Fare))+
    #scale_color_manual(values=c('orange','dark green'))+
    labs(title="Age and Fare",
         subtitle="Survival"))

## Scatter with regression

(MyS1<-ggplot(MyTitanicData, aes(x=Age, y=Fare, color=Sex, shape=Survived))+
    geom_point(aes(size=Fare)) + 
    geom_smooth(method=lm, se=FALSE, fullrange=TRUE)+
    scale_shape_manual(values=c(10, 15, 17))+ 
    scale_color_manual(values=c('orange','dark green', 'red'))+
    theme(legend.position="top")+
    labs(title="Survival"))

#################### DENSITY #######################
#################################################

(MyS3 <- ggplot(MyTitanicData,aes(x=Age, y=Fare, color=Survived)) + 
   geom_point() + 
   scale_color_manual(values = c('blue',"red")))

### Histogram
theme_set(theme_classic())

# Histogram on a Continuous (Numeric) Variable
(MyH1 <- ggplot(MyTitanicData, aes(Age)) + 
    scale_fill_manual(values=c('green',"dark red"))+
    geom_histogram(aes(fill=Survived), 
                   binwidth = 7, col="black") +  # col affect the outline
    labs(title="Histogram with Auto Binning", 
         subtitle="Age and Survival")) 

########################################
## Writing an image to a .jpeg
########################################
MyImageFileName="SomeImage.jpeg"
ggsave(MyImageFileName)