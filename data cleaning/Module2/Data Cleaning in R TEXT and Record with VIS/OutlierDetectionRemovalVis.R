########################################
##
##      Outliers
##
#########################################
## Gates
library(ggplot2)

filename="C:/Users/profa/Documents/R/RStudioFolder_1/DrGExamples/ANLY503/HeartRiskData_Outliers.csv"

HeartDF<-read.csv(filename)
head(HeartDF)

summary(HeartDF)
## Here - we get some hints...
## Max Cholesterol is 7000. We know this is not right.
## Max height is 93...this is curious only.
## Max weight is 500...this is curious....

## We have no min values that are suspect.

#######################
## SEE THE DATA!
##########################

########### Histograms...Bin size matters!
# Weight
(ggplot(HeartDF, aes(x=Weight,color=HeartDF$Gender, 
                     fill=HeartDF$Gender)) + 
  geom_histogram(binwidth=3))

#Height
(ggplot(HeartDF, aes(x=Height,color=HeartDF$Height, 
                     fill=HeartDF$Gender)) + 
    geom_histogram(binwidth=3))

#Cholesterol
(ggplot(HeartDF, aes(x=Cholesterol,color=HeartDF$Cholesterol, 
                     fill=HeartDF$Gender)) + 
    geom_histogram(binwidth=300))

############### Box plots......

(ggplot(HeartDF, aes(y=Weight,x="",
                     #color=HeartDF$Gender, 
                     fill=HeartDF$Label)) + 
    geom_boxplot())

#Height
(ggplot(HeartDF, aes(y=Height,x="", 
                     #color=HeartDF$Label, 
                     fill=HeartDF$Gender)) + 
    geom_boxplot())

#Cholesterol
(ggplot(HeartDF, aes(y=Cholesterol,x="", 
                     #color=HeartDF$Label, 
                     fill=HeartDF$Gender)) + 
    geom_boxplot())

#########################
## What do we see?
#########################

## Get the outliers... and view them...
outliers_C <- boxplot.stats(HeartDF$Cholesterol)$out
outliers_H <- boxplot.stats(HeartDF$Height)$out
outliers_W <- boxplot.stats(HeartDF$Weight)$out

(outlier_indices_C <- which(HeartDF$Cholesterol %in% c(outliers_C)))
(outlier_indices_H <- which(HeartDF$Height %in% c(outliers_H)))
(outlier_indices_W <- which(HeartDF$Weight %in% c(outliers_W)))

## Now we need the actual value at that index...
(Cout<-HeartDF$Cholesterol[c(outlier_indices_C )])
(HeartDF$Height[c(outlier_indices_H )])
(HeartDF$Weight[c(outlier_indices_W )])
Cout<-as.character(Cout)

(titleC<-paste("Cholesterol Levels with Potential Outliers at: ", 
               (paste(Cout, collapse = ", "))))

#Cholesterol
(ggplot(HeartDF, aes(y=Cholesterol,x="", 
                     #color=HeartDF$Label, 
                     fill=HeartDF$Gender)) + 
    geom_boxplot()+
    ggtitle("Cholesterol", subtitle = titleC)
  )

############# Lower and Upper Bounds
(lower_bound <- quantile(HeartDF$Cholesterol, 0.025))
(upper_bound <- quantile(HeartDF$Cholesterol, 0.975))

## Values below 1% or above 99% of the values....
lower_bound <- quantile(HeartDF$Cholesterol, 0.01)
upper_bound <- quantile(HeartDF$Cholesterol, 0.99)

outlier_indices <- which(HeartDF$Cholesterol < lower_bound | 
                           HeartDF$Cholesterol > upper_bound)
## rows, columns
HeartDF[outlier_indices, ]

#########################################
## Hampel Filter:
## ...outliers: values outside the interval 
## (I) formed by the median +/- 3 median absolute deviations 
##############################################################

(lower_bound <- median(HeartDF$Cholesterol) - 3 * mad(HeartDF$Cholesterol))

(upper_bound <- median(HeartDF$Cholesterol) + 3 * mad(HeartDF$Cholesterol))

(outlier_indicesH <- which(HeartDF$Cholesterol < lower_bound | 
                            HeartDF$Cholesterol> upper_bound))

HeartDF[outlier_indicesH, ]

################# --> interesting and not the same as the IQR

##############################
## 
## Grubbs's test
## Dixon's test - not ideal as sample size must be 3 - 30
## Rosner's test
##
##################################################

## Grubb
## Grubbs test detects one outlier at a time (highest or lowest value), so the null and alternative hypotheses are as follows:
##    H0: The highest (or lowest) value is *not* an outlier
##    H1: The highest (or lowest) value is an outlier

# install.packages("outliers")
library(outliers)
(GrubbsTest <- grubbs.test(HeartDF$Cholesterol))

# Results...
# Grubbs test for one outlier
# data:  HeartDF$Cholesterol
# G = 5.4680000, U = 0.0043895, p-value < 2.2e-16
# alternative hypothesis: highest value 7000 is an outlier

##  test for the lowest value, simply add the argument opposite = TRUE
(GrubbsTest_Lower <- grubbs.test(HeartDF$Cholesterol, opposite = TRUE))

## Grubbs by hand:
## https://www.statisticshowto.com/grubbs-test/
## https://www.graphpad.com/guides/prism/8/statistics/stat_detecting_outliers_with_grubbs.htm
## https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm


##################
##
## Rosner
##
#################
# Is used to detect several outliers at once (unlike Grubbs) 
#install.packages("EnvStats")
library(EnvStats)

(Rosner_test <- rosnerTest(HeartDF$Cholesterol,
                   k = 5  ## look for 5 potential outliers....
))


####################################
## FInal Vis option.........
######################################################
#install.packages("mvoutlier")
library(mvoutlier)

Y <- as.matrix(HeartDF[, c("Cholesterol", "Weight")])
#(res <- aq.plot(Y))
## This plot is SLOW...wait for it....
aq.plot(Y)

