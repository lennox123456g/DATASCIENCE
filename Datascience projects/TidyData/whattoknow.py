#tidy ata is a framework to structure data sets so that they can be easily analysed and visualized 
#its a goal one should aim for when cleaning data


#so each row is an observation 
#Each column is a variable 
#each type of observational unit forms a table 



#components of tidy data 
#each varaible must have its own column 
#each observation must have is own row
#Each value must have its own  cell

#Identify the components of tidy data
# Identify common data errors
# Usefunctions and methods to process and tidy data


#COLUMNS THAT HAVE VALUES INSTEAD OF VARIBLES A
# Datacanhavecolumnsthatcontainvaluesinsteadofvariables.Thisisusuallyaconvenient
# formatfordatacollectionandpresentation 


#HOW TO WORK WITH THEM 
#1. keep one column fixed 

#using dataonincomeandreligionintheUnitedStatesfromthePewResearchCenter
import pandas as pd 
pew = pd.read_csv('data/pew.csv')

# Note
# I usually use the terminology from the R world of using “pivot” to refer to going from
# wide data to long data and vice versa. I usually will specify the direction with “pivot
# longer” to go from wide data to long data, and “pivot wider” to go from long data to wide
# data.
# In this chapter “pivot longer” will refer to the dataframe .melt() method, and “pivot
# wider” will refer to the dataframe .pivot() method

#id_vars is a container (list, tuple, ndarray) that represents the variables that will
# remain as is.
# . value_vars identifies the columns you want to melt down (or unpivot). By default,
# it will melt all the columns not specified in the id_vars parameter.
# . var_name is a string for the new column name when the value_vars is melted
# down. By default, it will be called variable.
# . value_name is a string for the new column name that represents the values for the
# var_name. By default, it will be called value.

 # we do not need to specify a value_vars since we want to pivot
 # all the columns except for the 'religion' column

pew_long = pew.melt(id_vars='religion')
print(pew_long)


#melt method also existas as a pandas function 

#melt method 
pew_long = pew.melt(id_vars='religion')

#melt function 
pew_long = pd.melt(pew, id_vars='religion')


#we can name the melted/unpivoted column 
pew_leng = pew.melt( id_vars="religion", var_name="income", value_name="count")
print(pew_leng)

#keepig multiple clumns fixed(not meled )

 #usea listtoreferencemore than1variable
 billboard_long=billboard.melt(
 id_vars=["year","artist","track","time","date.entered"],
 var_name="week",
 value_name="rating",
 )
 print(billboard_long)

#Try with  Ebola data set 
ebola = pd.read_csv('data/country_timeseries.csv')
print(ebola.columns)
index(['Date', 'Day', 'Cases_Guinea', 'Cases_Liberia',
 'Cases_SierraLeone', 'Cases_Nigeria', 'Cases_Senegal',
 'Cases_UnitedStates', 'Cases_Spain', 'Cases_Mali',
 'Deaths_Guinea', 'Deaths_Liberia', 'Deaths_SierraLeone',
 'Deaths_Nigeria', 'Deaths_Senegal', 'Deaths_UnitedStates',
 'Deaths_Spain', 'Deaths_Mali'],
 dtype='object')

# print select rows and columns
 print(ebola.iloc[:5, [0, 1, 2,10]])

# The column names Cases_Guinea and Deaths_Guinea actually contain two variables.
# The individual status (cases and deaths, respectively) as well as the country name, Guinea.
# The data is also arranged in a wide format that needs to be reshaped (with the .melt()
# method).

#so lets melt the data into long format 
 ebola_long = ebola.melt(id_vars=['Date', 'Day'])
 print(ebola_long)

 #Splitting column individually 
 #can use split or .str accessor method 

# split() will split the string based on a  space,but
# we can pass in the underscore,_

#we use the .str accessor to canll the split method and then pass the _ underscore 


#getthe variablecolumn
#accessthe stringmethods
#andsplit thecolumnbasedon adelimiter

variable_split = ebola_long.variable.str.split('_')
print(variable_split[:5])

#the result is a list 
#now we want to split the different parts of the list and apply string methods 
 status_values = variable_split.str.get(0)
 country_values = variable_split.str.get(1)

 print(status_values)

#Now that we have the vectors we want, we can add them to our dataframe.
 ebola_long['status'] = status_values
 ebola_long['country'] = country_values

#USING EXPAND TO SPLIT  or can use pd.conctat() function
 #resetour ebola_longdata
ebola_long = ebola.melt(id_vars=['Date','Day'])

 #splitthe columnby_into adataframeusingexpand
variable_split = ebola_long.variable.str.split('_',expand=True)
print(variable_split)

#naming the new columns
ebola_long[['status','country']]= variable_split
print(ebola_long)


PIVOTING DATA IF a column has two variables
weather = pd.read_csv('data/weather.csv')
print(weather.iloc[:5, :11])

#melting the days
weather_melt = weather.melt(
 id_vars=["id", "year", "month", "element"],
 var_name="day",
 value_name="temp",
 )

 #pivoting the element 
weather_tidy = weather_melt.pivot_table(
    index = ['id', 'year', 'month', 'day'],
    columns = 'element',
    values = 'temp'
 )
 print( weather_tidy)


 #we can flatten the result 
weather_tidy_flat=weather_tidy.reset_index()
print(weather_tidy_flat)

#Applying the methods without an intermediate dataframe 
weather_tidy = (weather_mel)