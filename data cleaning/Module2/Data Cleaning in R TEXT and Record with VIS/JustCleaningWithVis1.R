###############################
##
## Data Visualization 
##
## Example 1: The Data Science
##             Life Cycle - 
##             Visually for RECORD DATA
##             with smaller dimensionality.
##
## Methods: cleaning, EDA, ML
##
## Gates
##
## Great reference:
## http://r-statistics.co/Top50-Ggplot2-Visualizations-MasterList-R-Code.html
##

## ASK YOURSELF THIS QUESTION:
## 
## Can you answer a question with a vis?

###################################

library(ggplot2)
##
## DATA SET
##
## https://drive.google.com/file/d/1KtnccHzbms1NGz2DzQVPM7W4U9s1s3bS/view?usp=sharing
##
##
Myfile="SummerStudentAdmissions3_.csv"
## USE YOUR OWN PATH - this is my path!
setwd("D:/Jieqian Liu/Data Science and Analystics/ANLY501 Data Science & Analytics/Module2/Data Cleaning in R TEXT and Record with VIS")

MyData <- read.csv(Myfile)


############################################
## Part 1: Cleaning the data
##         using data vis - ggplot
##
##         EDA is Exploratory Data ANalysis
##             Clean and explore...
################################################

## LOOK AT Each Variable.
str(MyData)
## Notice that there are 9 variables

## Variable (also called features, attributes, columns) Name
(MyVarNames<-names(MyData))
MyVarNames[1]
MyData[MyVarNames[1]]

(NumColumns <-ncol(MyData))


##############################
## Column 1: Decision
###################################

## THis is NOT part of the data!
## It is the LABEL of the data.

## Dataset labels should be of type factor
str(MyData$Decision)

## VISUALIZE to SEE what/where the errors are
theme_set(theme_classic())
MyBasePlot1 <- ggplot(MyData)
(MyBasePlot1<-MyBasePlot1 + 
    geom_bar(aes(MyData$Decision, fill = MyData$Decision)) + 
    ggtitle("Decision Label"))

## OK - I have problems. 
## 1) I have a blank level - likely from a missing value. 
## 2) I have a label called banana - which is wrong.

## Let's fix these.
## To fix factor data, first convert it to char. 
nrow(MyData)

MyData$Decision <- as.character(MyData$Decision)
## Keep only rows that are  "Admit", "Decline", or "Waitlist"

MyData <- MyData[(MyData$Decision == "Admit" | 
                    MyData$Decision == "Decline" | 
                    MyData$Decision == "Waitlist"),]

nrow(MyData)
## Check it again

(MyPlot1<-ggplot(MyData, aes(x=Decision, fill=Decision)) + 
    geom_bar()+
    geom_text(stat='count',aes(label=..count..),vjust=2)+
    ggtitle("Student Dataset Labels"))

## Good! Now we can see (and show others) that the
## Label in the dataset it clean and balanced.
## NOTE that we have color, a title, an x-axis label
## and labeled bars. We also have a legend.

## We are not done!!
str(MyData$Decision)
## This needs to be changed to type: factor
MyData$Decision<-as.factor(MyData$Decision)
## Check it
table(MyData$Decision)
str(MyData$Decision)
## Good! We now have factor data with 3 levels.

#################################################
## THe  next variable to look at is Gender
## Like Decision, Gender is also qualitative.
## Let's use a pie to look at it...
#################################################

str(MyData$Gender)
NumRows=nrow(MyData)
(TempTable <- table(MyData$Gender))
(MyLabels <- paste(names(TempTable), ":", 
                   round(TempTable/NumRows,2) ,sep=""))
pie(TempTable, labels = MyLabels,
    main="Pie Chart of Gender") 

#install.packages("plotrix")
library(plotrix)
pie3D(TempTable,labels=MyLabels,explode=0.3,
      main="Pie Chart of Gender ")


table(MyData$Gender)

## We have one problem. We have a blank or NA in the data
## We need to fix this. 

(sum(is.na(MyData$Gender)))  ## This confirms that it is not NA
## Let's look at str
str(MyData$Gender)
## This shows that we have blank and not NA....
## FIX - change to char, correct, change back to factor
## Keep track of what you are removing from the dataset
nrow(MyData)
MyData$Gender <- as.character(MyData$Gender)
## Keep only rows that are Male or Female

MyData <- MyData[(MyData$Gender == "Male" | 
                    MyData$Gender == "Female") ,]
nrow(MyData)
## Turn back to factor
MyData$Gender<- as.factor(MyData$Gender)
str(MyData$Gender)
table(MyData$Gender)

(TempTable <- table(MyData$Gender))
(MyLabels <- paste(names(TempTable), ":", 
                   round(TempTable/NumRows,2) ,sep=""))
pie(TempTable, labels = MyLabels,
    main="Pie Chart of Gender") 


############################################
## Next variable is: DateSub
#############################################
#names(MyData)
## Check format
str(MyData$DateSub)  ## It is incorrect.
## Check for NAs
(sum(is.na(MyData$DateSub)))
## Check the table
table(MyData$DateSub)
## The dates look ok - but the format is wrong and 
## needs to be DATE
(MyData$DateSub <- as.Date(MyData$DateSub, "%m/%d/%Y") )
str(MyData$DateSub)

## NOw that we have dates, can visualize them with 
## a time series vis option. 

ggplot(data = MyData, aes(x = DateSub, y = GPA))+
  geom_line(color = "#00AFBB", size = 2)

## We have a problem!
## The GPA should never be above 4.0. 

ggplot(MyData, aes(x = DateSub, y = GPA)) + 
  geom_area(aes(color = Gender, fill = Gender), 
            alpha = 0.5, position = position_dodge(0.8)) +
  scale_color_manual(values = c("#00AFBB", "#E7B800")) +
  scale_fill_manual(values = c("#00AFBB", "#E7B800"))

## We can already SEE many things. 
## We can see that Males applied a bit early and a bit later.
## We can see that we have an error in at least one GPA
## value that we will need to fix. 
## We can see that Female and Male application times and GPAs
## do not appear sig diff - but we can investigate this further.

#####################################################
##
##         Let's look at GPA and then dates with it
####################################################

str(MyData$GPA)  
MyData$GPA<-as.numeric(MyData$GPA)
table(MyData$GPA)

## Are there NAs?
(sum(is.na(MyData$GPA)))

## Fix the missing GPA first
## Find it
(MissingGPA <- MyData[is.na(MyData$GPA),])
## OK - its a Female/Admit. We can replace the missing GPA
## with the median of all Female Admits.
(Temp<-MyData[MyData$Decision=="Admit" & MyData$Gender=="Female",])
## The median for Female Admits is:
(MyMed<-median(Temp$GPA, na.rm=TRUE))
## NOW - replace the missing GPA with this Median
MyData$GPA[is.na(MyData$GPA)] <- MyMed
## Check to assure the missing value  was updated...
(sum(is.na(MyData$GPA)))

table(MyData$GPA)

## While I will use plyr below - it is good to 
## know how to use functions as well...
##-------------------------------------------
## Create a function to get medians
GetMedian <- function(x){
  out<-median(x)
  return(out) 
}

## Check and use the function
(MaleMed<-GetMedian(MyData$GPA[which(MyData$Gender=="Male")]))
(FemaleMed<-GetMedian(MyData$GPA[which(MyData$Gender=="Female")]))

##---------------------------------------------

library(plyr)

## Create a table using the dataset
## This table is BY Gender
## The method is summarize
## A new column is med and is the median for GPA
(TEMPmeds <- ddply(MyData, .(Gender), summarize, 
                   med = median(GPA)))


## Next, we have an incorrect value....let's SEE IT

(MyV1 <- ggplot(MyData, aes(x=Gender, y=GPA, fill=Gender)) + 
    geom_violin(trim=TRUE)+ geom_boxplot(width=0.1)+
    geom_text(data = TEMPmeds, 
              aes(x = Gender, y = med, label = med), 
              size = 3, vjust = -1.5,hjust=-1)+
    ggtitle("GPA and Gender")+
    geom_jitter(shape=16, position=position_jitter(0.2)))

## Now we can SEE the issue. There is at least one GPA
## that is out of range. Let's fix this.
## Let's replace the missing GPA by finding the median
## for the ADMITS in that Gender group

## FIND the row with GPA > 4
(WrongGPAs <- MyData[(MyData$GPA<0 | MyData$GPA >4),])
## 
## We have Male Admit with a GPA of 6.

## Fix it by using Male Admit GPA Median
(Temp<-MyData[MyData$Decision=="Admit" & MyData$Gender=="Male",])
## The median for Male Admits is:
(MyMed<-median(Temp$GPA, na.rm=TRUE))
## NOW - replace the missing GPA with this Median
MyData$GPA[MyData$GPA>4] <- MyMed

## NOW VISUALIZAE IT AGAIN:
(TEMPmeds <- ddply(MyData, .(Gender), summarize, 
                   med = round(median(GPA),2)))


## Next, we have an incorrect value....let's SEE IT

(MyV1 <- ggplot(MyData, aes(x=Gender, y=GPA, fill=Gender)) + 
    geom_violin(trim=TRUE)+ geom_boxplot(width=0.1)+
    geom_text(data = TEMPmeds, 
              aes(x = Gender, y = med, label = med), 
              size = 4, vjust = -2.5,hjust=-1.8)+
    ggtitle("GPA and Gender")+
    geom_jitter(shape=16, position=position_jitter(0.2)))

## That's better!

table(MyData$GPA)
## LOOKS GOOD!


#############################################
##
##            Let's look at State next
############################################
#names(MyData)
str(MyData$State)
## Let's use a BAR to look
BaseGraph <- ggplot(MyData)
(MyG3<-BaseGraph + 
    geom_bar(aes(State, fill = Gender), position="dodge")+
    ggtitle("States and Gender"))

## UGLY!!

## Let's make this nicer so we can READ THE X AXIS
(MyG3<-BaseGraph + 
    geom_bar(aes(State, fill = Gender), position="dodge")+
    ggtitle("States and Gender")+
    theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)))

## MUCH BETTER!

## Now we can SEE that we have problems :)
## First, we have poor balance. It might be needed to 
## collect all the lower count states, such as ALabama, Mississippi, 
## etc. into a group called OTHER. However, we will not do this here.
## If you want to see how - look at this other tutorial
## http://drgates.georgetown.domains/SummerClassificationRMarkdown.html

## Also - We have two Virginias  - we need to combine them:

MyData$State[MyData$State == "virginia"] <- "Virginia"
table(MyData$State)
## Now - we need to remove the level of virginia
MyData$State<-as.character(MyData$State)
table(MyData$State)
MyData$State<-as.factor(MyData$State)
str(MyData$State)


## Check it
(MyG4<-ggplot(MyData) + 
    geom_bar(aes(State, fill = Gender), position="stack")+
    ggtitle("States and Gender")+
    theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)))

## Even better!

#########################################
##
## Now let's look at WorkExp
#######################################
#names(MyData)
(sum(is.na(MyData$WorkExp)))
str(MyData$WorkExp)

## Let's look
theme_set(theme_classic())

# Histogram on a Continuous (Numeric) Variable
(MyS3 <- ggplot(MyData,aes(x=WorkExp, y=GPA, color=Decision)) + 
    geom_point() + 
    scale_color_manual(values = c('blue',"red", "green")))

## This helps in many ways. We can see that we have no outliers
## or odd values. 
## However, let's check it with a box plot. 

(MyL1<-ggplot(MyData, aes(x=Decision, y=WorkExp))+
    geom_boxplot()+
    geom_jitter(position=position_jitter(.01), aes(color=Gender))+
    ggtitle("Work Experience, Admissions, and Gender"))

## This looks good and it also starts to tell us that people
## were not penalized or prefered based on work experience.

#####################################################
##
##          Let's look at TestScore and Writing Score
##      
#######################################################
(sum(is.na(MyData$TestScore)))
(sum(is.na(MyData$WritingScore)))

str(MyData)

## Box plots are great to look for odd values

(MyL2<-ggplot(MyData, aes(x=Decision, y=TestScore))+
    geom_boxplot()+
    geom_jitter(position=position_jitter(.01), aes(color=Gender))+
    ggtitle("Test Score, Admissions, and Gender"))

## Interesting!! This mostly makes sense except for the 800
## in the Admit group. However, it is not an outlier - it is
## just interesting.

(MyL3<-ggplot(MyData, aes(x=Decision, y=WritingScore))+
    geom_boxplot()+
    geom_jitter(position=position_jitter(.01), aes(color=Gender))+
    ggtitle("Writing Score, Admissions, and Gender"))

## Hmmm - most of this looks OK, BUT, we have some very strange
## values for the Admit group. 

## *** Let's look at these:


(Temp <- subset(MyData, Decision=="Admit", 
                select=c(Decision,WritingScore)) )
table(Temp$WritingScore)

## OK - we can see that two score seem incorrect. 
## The 1 and the 11, for an Admit, it not likely. 
## Let's replace them with median

(Temp3<-MyData[MyData$Decision=="Admit",])
## The median for Admits is:
(MyMed2<-median(Temp3$WritingScore, na.rm=TRUE))
## NOW - replace the incorrect  with this Median
MyData$WritingScore[MyData$WritingScore<85] <- MyMed2


## check again
(MyL4<-ggplot(MyData, aes(x=Decision, y=WritingScore))+
    geom_boxplot()+
    geom_jitter(position=position_jitter(.01), aes(color=Gender))+
    ggtitle("Writing Score, Admissions, and Gender"))

## MUCH BETTER!

## We can also look using density area plots...


# Use semi-transparent fill
(MyPlot4<-ggplot(MyData, aes(x=WritingScore, fill=Decision)) +
    geom_area(stat ="bin", binwidth=2, alpha=0.5) +
    theme_classic())

## Here - using density - we can get a deeper look
MyPlot5 <- ggplot(MyData, aes(WritingScore))
MyPlot5 + geom_density(aes(fill=factor(Decision)), alpha=0.5) + 
  labs(title="Density plot", 
       subtitle="Decisions Based on Writing Scores")


### Hmmm - does it seem like WritingScore is really
## related to Admissions?

## Let's run an ANOVA test to see
MyANOVA_WS_Adm <- aov(WritingScore ~ Decision, data = MyData)
# Summary of the analysis
summary(MyANOVA_WS_Adm)  ## The test IS significant!
plot(MyANOVA_WS_Adm, 1)
## The above shows we can assume the homogeneity of variances.
plot(MyANOVA_WS_Adm, 2) ## Close to normal

library("ggpubr")
ggboxplot(MyData, x = "Decision", y = "WritingScore", 
          color = "Decision", palette = c("#00AFBB", "#E7B800","green"),
          ylab = "WritingScore", xlab = "Decision")


## Let's add labels...

(TheMean <- ddply(MyData, .(Decision), summarize, 
                  mean2 = round(  mean(WritingScore) ,2 )))


## Another View...

(MyV2 <- ggplot(MyData, aes(x=Decision, y=WritingScore, fill=Decision)) + 
    geom_violin(trim=TRUE)+ geom_boxplot(width=0.1)+
    geom_text(data = TheMean, 
              aes(x = Decision, y = mean2, label = mean2), 
              size = 3, vjust = -1.5,hjust=-1)+
    ggtitle("Writing Score and Admissions Decision")+
    geom_jitter(shape=16, position=position_jitter(0.2)))

###########################################
##  The last variable is VolunteerLevel
##  
##############################################
str(MyData$VolunteerLevel)
## This should NOT be an int
## COrrect it to factor
MyData$VolunteerLevel <- as.factor(MyData$VolunteerLevel)
table(MyData$VolunteerLevel)

(MyG1<-ggplot(MyData) + 
    geom_bar(aes(VolunteerLevel, fill = Decision)) + 
    ggtitle("Decision by Volunteer Level")+
    coord_flip())
