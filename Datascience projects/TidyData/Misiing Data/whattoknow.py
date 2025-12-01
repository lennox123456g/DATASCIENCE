#rarely you will be given a datset without missing values 
#recognise potential ways data can go missing in data processing 
#use different functions to fill in missing values 


#to test for missing value in pandas 

#Missing values come from datasets with missing values or data munging processses 
#we use isnull and notnull

# NA, NaN, or nan are the default missing values 

import pandas as pd 
print(pd.isnull(NaN))
print(pd.isnull(NAN))
print(pd.isnull(nan))

print(pd.notnull(42))  #returns true 

print(pd.notnull('missing'))  #returns true 

#To set missing values in 0our data set 
#we use keep_default_na , , na_filter and na_values which takes a list 
#and note that these methods are available under the read_CSV FUNCTION
#sometimes we do thes ethings for perfomance boost while working with data 

#Example 

#set location of the data 
visited_file = 'data/survey_visited.csv'
print(read_csv(visited_file))

#now to put missing values 
print(pd.read_csv(visited_file, keep_default_na=False))

print(pd.read_csv(visited_file, na_values=[""], keep_default_na=False))


#MISSING VALUES IN MERGED DATA 
visisted = pd.read_csv('data/survey_visited.csv')
survey = pd.read('data/survey_survey.csv')

print(visited)
print(survey)

vs = visited.merge(survey, left_on='indent', right_on='taken')
print(vs)


#CREATIG MISSING VALUES VIA SERIES AND THE DATAFRAME INPUT 
#Misising value in series 
num_legs = pd.series({'goat': 4, 'amoeba': nan})
print(num_legs)


#missing values in dataframe 
Scientists = pd.DataFrame(
    {
        "Name": ["Rosaline Franklin", "William Gosset"],
        "Occupation": ["Chemist", "STatistician"],
        "Born":  ["1920-07-25", "1876-06-13"],         
        "Died": ["1958-04-16", "1937-10-16"],
        "missing": [NaN, nan],
    }
)
print(Scientists)
#You will notice the dtype of the missing column will be a float64. This is because the
# NaN missing value from numpy is a floating point value.


#REINDEXING 

#IF ONE WANTS TO ADD NEW INDICES TO YOUR DATAFRAME 
#but yet they want to retain its original values

gapminder = pd.read_csv('data/gapminder.tsv', sep='\t')

life_exp = gapminder.groupby(['year'])['lifeExp'].mean()
print(life_exp)


#now lets reindex the data by subsetting the data and using reindex() method

#subset
y2000 = life_exp[life_exp.index > 2000]
print(y2000)

#reindex
print(y2000.reindex(range(2000,2010)))


#KNOWING THE NUMBER OF MISSING VALUES USING COUNT 

ebola = pd.read_csv('data/country_timeseries.csv')

#count the number of  non-missing values 
print(ebola.count())



# we can subtract number of non missing rows from number of rows 
num_rows = ebola.shape[0]
num_missing = num_rows - ebola.count()
print(num_missing)


#TO KNOW THE NUMBER OF MISSING WE CAN USE COUNT_NON ZERO OF NUMPY TOGETHER WITH .IS_NULL METHOAD
import  numpy as np
print(np.count_nonzero(ebola.isnull()))


print(np.count_nonzero(ebola['Cases_Guinea'].isnull()))

#METHOD 2 IS USING .VALUE_COUNTS() METHOD ON A SERIES 
#WHICH PRINTS FREQUESCTY TABLE OF VALUES 
#s. If you use the dropna parameter, you can
# also get a missing value count.

#Value counts form the Cases_Guinea column
cnts = ebola.Cases_Guinea.value_counts(dropna = False)
print(cnts)

#to select the values missing 
print(cnts.loc[pd.isnull(cnts.index)])


#checkif thevalueismissing, andsumuptheresults
 print(ebola.Cases_Guinea.isnull().sum())

 #WHAT TO DO FOR MISSING DATA 
#fill it with another value 
#fill with existing data 
#drop data from the dataset

#option one filling with anotehr value using fillna 
#fill the missing value to 0 and only look at the first 5 columns 
print(ebola.fillna(0).iloc[:, 0:5])

#FORWARD FILL
# We can use built-in methods to fill forward or backward.When wef ill dataforward, the
# last known value(fromtoptobottom)isusedforthenextmissingv alue.Inthisway,
# missingv aluesarereplacedwiththel ast known and recordedv value.
print(ebola.fillna(method = 'ffill').iloc[:, 0:5])

#if a column begins with a missing value ,then the dat will remain missing because  there is no previous value to fill in 


#BACKWARD FILL 
# We can alsohave Pandas fill data backward.When we fill data backward, the newest value
# (from top to bottom)isused to replace the missing data.In this way,missing values are
# replaced with the newest value.
print(ebola.fillna(method='bfill').iloc[:, 0:5])

#if a column ends with a missing value ,then the dat will remain missing because  there is no previous value to fill in

#INTERPOLATION 
Interpolation iuese existing values to fill in missing values .It feels in missing values linearly, Specifically it treats the missing values a if they should be equally spaced 
print(ebola.interpolate().iloc[:, 0:5])

# Noticehowitbehaveskindof inaforwardfill fashion,butinsteadofpassingonthelast
# knownvalue, itwill fill inthedifferencesbetweenvalues.
# The.interpolate()methodhasamethodparameterthatcanchangetheinterpolation
# method.1PossiblevaluesatthetimeofwritinghavebeenreproducedinTable9.1.


#DROPPING MISSING VALUES 
#Thelastwaytoworkwithmissingdataistodropobservationsorvariableswithmissing
# data.Dependingonhowmuchdataismissing,keepingonlycompletecasedatacanleave
# youwithauselessdataset.Perhapsthemissingdataisnotrandom,sothatdropping
# missingvalueswill leaveyouwithabiaseddataset,orperhapskeepingonlycompletedata
# will leaveyouwithinsufficientdatatorunyouranalysis.
#  We can use the.dropna() method to drop missing data, and specify parameters to this
# method that control how data are dropped. For instance, the how parameter lets you specify
# whether a row (or column) is dropped when 'any' or 'all' of the data is missing. The
# thresh parameter lets you specify how many non-NaN values you have before dropping the
# row or column.
print(ebola.shape)

#If we keep only complete cases in our Ebola data set, we are left with just one row
# of data.

ebola_dropna = ebola.dropna()
print(ebola_dropna.shape)
print(ebola_dropna)


#CALCULATIONS WITH MISSING DATA 
#nan value in calculaton automaticallyhave a skipna parmaeter 
#that will calculate a value by skipping over the  missing values 

 #skipping missing values is  True by default
 print(ebola.Cases_Guinea.sum(skipna=True))