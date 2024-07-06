# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:26:37 2018

@author: profa
"""

### 
### Data Exploration
### Data Cleaning
### Data Processing
###
### Gates

### This examples uses the Kaggle Titanic Training dataset


import pandas as pd
import numpy as np
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt


## Data *always* needs to be cleaned and prepared
## Create an account with Kaggle so you can login
## Source: https://www.kaggle.com/sudhirtk/titanic-train-dataset

## Once you download the data - save it as .csv
## Use a smart name. I used Titanic_Kaggle_Train_Data.csv
## Make sure you can see the dataset to the right under File explorer

##### Bring the data in using pandas
filename="Titanic_Kaggle_Train_Data.csv"
## NOtice that I did not hard-code the filename
TitanicDF = pd.read_csv(filename)
## Check to see if this worked
print(type(TitanicDF))
print(TitanicDF.head(15))

## I notice right away that Cabin has a lot of 
## NaN values *and* does not contain information
## that I need. I also do not want to keep Ticket.

############  DROP a COLUMN

drop_these = ["Cabin", "Ticket"]

TitanicDF.drop(drop_these, inplace=True, axis=1)
## axis = 0 are the rows

## Let's see what this did - you will see that we now have only 10 cols
print(TitanicDF.head(15))

## We can also use the column number
## The "Name" column is column 3. Recall that Python starts at 0
## Let's drop the Name column as well
cols = [3,7]
TitanicDF.drop(TitanicDF.columns[cols], inplace=True, axis=1)
print(TitanicDF.head(15))
## Now we have 8 columns remaining

##############  GET THE TYPES OF EACH COLUMN/VARIABLE
print(TitanicDF.dtypes)

## The next steps include looking at each variable individually
## We will determine if it has the type we want, or if it needs 
## to change. 

### PassengerId has type int64
### An ID is not numeric and so should not be an integer type.
### We can change this to character as it is a name of sorts
### but *NOT* a category.
## Do not forget - you must asign the df to the new type
TitanicDF["PassengerId"]= TitanicDF["PassengerId"].astype(str)
print(TitanicDF.dtypes)

### The "Survived " column is very important. Why?
## This column offers a *label* for this dataset
## Because this is labeled data, supervised methods can 
## be applied to it later. 

###          !!!!!!!!!! IMPORTANT !!!!!!!!!!!!     ##

## To use a label to train any ML methods, the label should
## be a "factor" / "category" data type. Not numeric or char, etc.

## So, let's change "Survived" from int64 into a factor
## NOTES:
## Categoricals are a pandas data type 
## corresponding to categorical variables in statistics.
## A categorical variable takes on a limited, and usually fixed, 
## number of possible values (categories; levels in R)
## All values of categorical data are either in categories or np.nan

TitanicDF["Survived"] = TitanicDF["Survived"].astype('category')
print(TitanicDF.dtypes)

## Now, we can move a bit faster....let's find all the variables
## that should be categories and change them all at once
TitanicDF["Sex"] = TitanicDF["Sex"].astype('category')
TitanicDF["Pclass"] = TitanicDF["Pclass"].astype('category')
TitanicDF["Embarked"] = TitanicDF["Embarked"].astype('category')
print(TitanicDF.dtypes)

### Next, I see that "Age" is a float64 or 64 bit decimal value
## This is fine. Age is a numeric value, as is SibSp and Fare
## So - the data types have now been successfully corrected.

#########     DEALING WITH MISSING VALUES #################

## Missing values find there way into many places.
## A good first step is to explore  - see how many are there

## Get a list of all the column names
ColumnNamesList = TitanicDF.columns.values
print(ColumnNamesList)

## Print the number of missing values in all variables
for name in ColumnNamesList:
    print(name, ": ")
    total_nas=TitanicDF[name].isna().sum()
    print(total_nas)
    
## Remember - you can double check anything you code
## by creating a small dataset and seeing if it works
    
#### EXAMPLE
Small=pd.read_csv("SmallExampleDataWithMissingValues.csv")
print(Small)
## Results: this replaced NaN, NA, and blanks with NaN

## So - back to the missing data in our Titanic dataset
## Let's again look at one variable at a time

## From the results, we see that PassengerId, Pclass, Survived, 
## Sex, and Fare have no missing values. That is very good and 
## unusual. 

## However, "Age" has 177 missing values! That's a lot
print(TitanicDF.shape)
## So I have 891 rows and 8 columns

## This means that 177 NA values is 177/891 or about 20%
## We have some options.
## First, we can remove all the rows with missing values. 
## If we do this, we lose about 20% of the data - not great!
## We can replace the values with a mean or median.
## This is risky - especially if Age is critical to the 
## analysis. 

## Because this is a tutorial - I will offer two examples:
## 1) We will remove all rows for which "Age" is NaN
## 2) We will replace the NaN values under Embark with the *mode*

### FIrst - update the NaNs under Embark to be the mode
print(TitanicDF.Embarked)
TheMode=stat.mode(TitanicDF.Embarked)
print(TheMode)

## Pandas provides the fillna() function for replacing missing 
## values with a specific value.
TitanicDF.Embarked.fillna(TheMode, inplace=True)
## Count the NaN's again...
print(TitanicDF.isnull().sum())

## OK - so far so good. Now we need to worry about the 177 NaN
## values under the Age column.

## In some cases, one might choose to retain these rows and 
## replace the missing values with the mean or median.
## However, Titanic data (and survival) is correlated to Age
## and we do not want to lose that!

## So, we will drop all row with NaN
TitanicDF.dropna(inplace=True)
## Check it now
print(TitanicDF.isnull().sum())

#################   INCORRECT VALUES ######################
## The steps above are often easier. 
## Incorrect values are harder to find and harder to fix
## As a first step, let's explore the data

#### Step 1 - Look at each variable and some basics stats and vis
print(TitanicDF.columns.values)
print(TitanicDF.describe())
plot1=sns.boxplot(x="Age",data=TitanicDF)
plt.show()
plot2=sns.swarmplot(x="Age",  data=TitanicDF, color=".25")
plt.show()
## To get plots to pop up and not be inline (the reverse...)
## Tools --> Preferences --> IPython --> Graphics --> Automatic

## Look at the min for Age!
## This is .42. That means there are some incorrect values and
## we will need to remove those rows.

## Remove any ages below 1 
## Because the max Age is 80  - we are OK there

TitanicDF=TitanicDF[TitanicDF.Age > 1]
print(TitanicDF.head(15))
print(TitanicDF.describe())

## Look at all of the other variables to see if there are any issues

## The Fare is interesting...the max is very large - let's 
## use a boxplot to take a closer look
plot3=sns.boxplot(x="Fare",data=TitanicDF)
plt.show()
## We have some outliers!
## Let's count up the Fares that many deviations above the mean
## The "many" is subjective. I am playing safe

print(TitanicDF[TitanicDF.Fare > 400 ].count() )
## There are three Fares that are above $400 - let's remove these rows
TitanicDF=TitanicDF[TitanicDF.Fare < 400]
print(TitanicDF.describe())
## This is good. We only removed three rows and the new max is 
## almost half the size it was before

## Our SibSp looks OK
## Let's make some plots for the non-numeric variables....
print(TitanicDF.head(15))
sns.countplot(x="Survived",data=TitanicDF)
plt.show()

sns.countplot(x="Pclass",data=TitanicDF)
plt.show()

sns.countplot(x="Sex",data=TitanicDF)
plt.show()

































