#CONVERSION OF DATATYPES usimg functions
#modify categorical data 
#identify the datatype in a column 


#using built in tips dataset 
import pandas as pd 
import seaborn as sns 
tips = sns.load set("tips")

#getiing the list of datatypes in the columns 
print(tips.dtypes)

#converting t strings 
.astype( newdatatype here )

#convert tecategory sex column into a string datatype 
tips['sex_str'] = tips['sex'].astype(str)

#using astype again 

#convert totl_ ill into a string 
tips ['total_bill'] = tips['total_bill'].astype(str)
print(tips.dtypes)


#cnverting it back to float 
tips['total_bill'] = tips['total_bill'].astype(float)

#USING TO_NUMERIC METHOD
#it handles non numeric values better 
#This can be used in case instead of Nan value that represents a misssing valuer in pandas , a umeric colum might
#might use the string misssing  or 'null' for  that purpose instead 
#Rhis will make the entire  column a string type  instead of a  numeric type

#Example 
#putting in missing value in out data to experiment 
#subset the tips data 
tips_sub_miss = tips.head(10).copy()
# Weusethe.copy()methodheretoavoidtheSettingWithCopyWarningmessagewhenwe
# modifythesubsetteddataset(

print(tips_sub_miss)

#when we try using astype() it fails because pandas doesnt know missing datatype

# this will cause an error 
tips_sub_miss['total_bill'].astype(float)


 # this will cause an error
 pd.to_numeric(tips_sub_miss['total_bill'])

#also to_ unumeric alone doesnt work , we need to add itd  coerce parameter that chance missing value to Nan
tips_sub_miss["total_bill"]= pd.to_numeric(
 tips_sub_miss["total_bill"],errors="coerce"
 )

# ‘raise’, then invalid parsing will raise an exception
#‘coerce’, then invalid parsing will be set as NaN
# ‘ignore’, then invalid parsing will return the input
print(tips_sub_miss.dtypes)



#CATEGORICAL FATA 
#‘raise’, then invalid parsing will raise an exception
#‘coerce’, then invalid parsing will be set as NaN
 #‘ignore’, then invalid parsing will return the input


 #COnverting colmn to categorical data 

#Convert the sex column into a string object first 
tips['sex'] = tips['sex'].astype('str')
print(tips.info())

 # convert the sex column back into categorical data
 tips['sex'] = tips['sex'].astype('category')
 print(tips.info())

 #CATEGORICAL DATA HAS METHODS TO WORK ON IT 


 #STRINGS AND TEXT DATA 
 #here are two strings 
 word = 'grail'
 sent = 'a scratch'

 #subsetting and slicing these strings 
 prnt(word[0])
 print(sent[3])

 #slicing multiple letters 
 #getting the first three letters 
 print(word[0:3])

 #getting the last number using negative index 
 print(senr[-1])

 # get 'a' from "a scratch"
 print(sent[-9:-8])


 #last letter cant be got when its index is used as a second value ]
 #sctch
 print(sent[-7:-1])

 #we use te len 
  # note that the last index is one position is smaller than
 # the number returned for len
 s_len = len(sent)
 print(s_len)
 print(sent[2:s_len])
# scratch


 print(word[0:3])
# gra
 # left the left side empty
 print(word[ :3])
# gra
 print(sent[2:len(sent)])
 scratch
 # leave the right side empty
 print(sent[2: ])
# scratch

# Another way to specify the entire string is to leave both values empty.
 print(sent[:])

 #increment by 2
 # get every third character
 print(sent[::3])

 #there are many string methods 


"black Knight".capitalize()
 "It's just a flesh wound!".count('u')
 "Halt! Who goes there?".startswith('Halt')
 "coconut".endswith('nut')
 "It's just a flesh wound!".find('u')
 "It's just a flesh wound!".index('scratch')
 "old woman".isalpha()
 "37".isdecimal()
 "I'm 37".isalnum()
 "Black Knight".lower()
 "Black Knight".upper()
"flesh wound!".replace('flesh wound', 'scratch') 
 " I'm not dead. ".strip()
 "NI! NI! NI! NI!".split(sep=' ')
 "3,4.partition(',')
 "nine".center(width=10)
 "9".zfill(with=5)


 #JOIN METHOD to join strings 
 d1 = '40 "'
 m1 = "46'"
 s1 = '52.837"'
 u1 = 'N'

 d2 = '73°'
 m2 = "58'"
 s2 = '26.302"'
 u2 = 'W'

 #we want to join them with a space ""
 #we use the join() method

 coords = "".join([d1, m1, s1, u1, d2, m2, s2, u2])
 # answer 40° 46' 52.837" N 73° 58' 26.302" W


 #we cpuld as we;; split on the '' into individual parts 
 coords.split("")
 ['40°', "46'", '52.837"', 'N', '73°', "58'", '26.302"', 'W']


 #split lines 
 #retiurns  a lsit of elements each on its line 
 multi_str = """Guard: What? Ridden on a horse?
 King Arthur: Yes!
 Guard: You're using coconuts!
 King Arthur: What?
 Guard: You've got ... coconut[s] and you're bangin' 'em together.
 """
 print(multi_str)


multi_str_split = multi_str.splitlines()
print(multi_str_split)


#If you want only the message from the guard 
guard = multi_str_split[::2]#keep getting the second interal

#if we getting fromm the string 
# There are a few ways to just get the lines from the “Guard.” One way would be to use
# the .replace() method on the string and .replace() the Guard: string with an empty
# string ''. We could then use the .splitlines() method

 guard = multi_str.replace("Guard: ", "").splitlines()[::2]
 print(guard)


 #STRING FORMATTING
s = f"hello"
 print(s)

#  This tells the string that it is an f-string. This now allows us to use { } within the string
# to put in Python variables or calculations

 num = 7
 s = f"I only know {num} digits of pi."
 print(s)

  const = "e"
 value = 2.718
 s = f"Some digits of {const}: {value}"
 print(s)
 Some digits of e: 2.718
 lat = "40.7815° N"
 lon = "73.9733° W"
 s = f"Hayden Planetarium Coordinates: {lat}, {lon}"
 print(s)

# Hayden Planetarium Coordinates: 40.7815° N, 73.9733° W
# Variables can be reused within a f-string.
 word = "scratch"
 s = f"""Black Knight: 'Tis but a {word}.
 King Arthur: A {word}? Your arm's off!
 """
 print(s)


#FORMATTING NUMBERS 
#WITH COMMAS ETC 


#REGULAR EXPRESSIONS 


#DATES AND TIMES 
#One of the bigger reasons for using Pandas is its ability to work with timeseries data.
#Pythonns datetime object from the atetime library 
from datetime import datetime 

#lets use it to get the current dattime 
now = datetime.now()
print(f"Last time this chapter was rendered for print: {now}")

# Last time this chapter was rendered for print: 2022-09-01 01:55:41.496795

#creatining our own datetime manually
t1 = datetime.now()
t2 = datetime(1970, 1, 1)


# And we can do datetime math
 diff=t1-t2
 print(diff)

# The data type of a date calculationis a timedelta.

 print(type(diff))


 #CONVERTING TO DATETIME 
 #using the ebola datset 
 #USING TO_DATETIME 
 import pandas as pd 
 ebola = pd.read_csv('data/country_timesries.csv')

 #top left corner of the data 
 print(ebola.iloc[:5,:5])

 print(ebola.info())

 #create a new column that converts the dae column into datetime 
 ebola['date_dt'] = pd.to_datetime(ebola['Date'])

# Theto_datetime()functionhasaparametercalledformatthatallowsyoutomanually
# specifytheformatofthedateyouarehopingtoparse.Sinceourdateisina
# month/day/yearformat,wecanpassinthestring%m/%d/%Y.


 ebola['date_dt']= pd.to_datetime(ebola['Date'],format='%m/%d/%Y')

print(ebola.info())

# strptime could be looked into 


#loading dat with dates 
#CSV DATA TAKES A FUNCTION THAT ALLOWS FOR THHAT 
# it HAS mnany parameters ,–forexample,parse_dates,
# inher_datetime_format,keep_date_col,date_parser,dayfirst,and cache_dates.

#this can be done automatically at the start of loading data 
 ebola= pd.read_csv('data/country_timeseries.csv', parse_dates=["Date"])
 print(ebola.info())

 #EXtracting the date components 
 d = pd.to_datetime('2021-12-14')
  print(type(d))

print(d.year)
 2021
 print(d.month)
 12
 print(d.day)
 14

# We can create a new year column based on the Date column.
 ebola['year'] = ebola['date_dt'].dt.year
 print(ebola[['Date', 'date_dt', 'year']])

 #PARSING THE DATE 
ebola= ebola.assign(
    month=ebola["date_dt"].dt.month,
    day=ebola["date_dt"].dt.day
 )
 print(ebola[['Date','date_dt','year','month','day']])

 print(ebola.info()) #DATASET IS NOT Preserved 


 #CALCULATIO IN DATE 


 #Example aerlies ate - minimum date 
 print(ebola.iloc[-5:, :5])

 #earliest date 
 print(ebola['date_dt'].min())

# We can use this date in our calculation.
 ebola['outbreak_d'] = ebola['date_dt']- ebola['date_dt'].min()
 print(ebola[['Date', 'Day', 'outbreak_d']])

 print(ebola.info())


 #DATETIME METHODS 
 #BANK FAILURES DATASET 
 banks = pd.read_csv('data/banklist.csv')
 print(banks.head())

 #calling the date with the dates directly parsed 
 banks = pd.read_csv(
    "data/banklist.csv",parse_dates = ["closing Date", "Update Date"]
 )

print(banks.info())

# We can parse out the date by obtaining the quarter and year in which the bank closed.

banks = bank.assign(
    closing_quarter = banks['Closing Date'].dt.quarter,
    closing_year = banks Closing Date .dt.year
)

closing_year banks.groupby closing_year .size()

closing _year_q = (
    anks
    .groupby(['closing_year', 'closing_quarter'])
    .size()
)

#NOW WE CAN PLOT THESE RESULTS 
import matplotlib.pyplot as plt 

fig, ax = plt.subplots()
ax = closing_year.plot()
plt.show()

fig, ax = plt.subplots()
ax = closing_quarter.plot()
plt.show()


#GETTING STOKE DATA

#pandas uses the pandas-datreader library automatically 

#we can install adn used the pandas_datafreader
#to get data from the internet 
import pandas_datafreader.data as web 

#in this rxample we are gettting stock information about Tesla
tesla = web.DataReader('TSLA','yahoo')

print(tesla)


#the stock data was saved 
#so we dont need to rely on the internet again 
#instead we can load the same dataset as a file 
tesla = pd.read_csv(
    'data/tesla_stock_yahoo.csv', parse_dates=["Date"]
)

print(tesla)



#BOOLLEAN SUBSETTING DATES 
#wecan
# incorporatethesemethodstosubsetourdatawithouthavingtoparseouttheindividual
# componentsmanually.
print(
    tesla.loc[
        (tesla.Date.dt.year == 2010) & (tesla.Date.dt.month == 6)
    ]
)

#DATETIME ALWAYS SET AS THE DATAFRAME INDEX
tesla.index = tesla['Date']  #DatetimeINdex Object
print(tesla.index)


#With the index set as a date object, we can now use the date directly to subset rows. For
# example, we can subset our data based on the year.
 print(tesla['2015'])


print(tesla.loc['2015'])

#Alternatively, we can subset the data based on the year and month.
print(tesla.loc['2010-06'])

#TIMEDELTA timedelta to create a TimedeltaIndex.

tesla[ref_date ] = tesla['Date'] - tesla['Date'].min()

#now we can assiggn the timedelta to the index 
tesla.index = tesla['ref_date']

print(tesla)

print(tesla['0 day': '10 day'])


#DATA RANGES 
#Not every data set will have a fixed frequency of values. F

 ebola = pd.read_csv(
    'data/country_timeseries.csv', parse_dates=["Date"]
 )

# We’ll justworkwiththefirstfiverowsinthisexample.
 ebola_5=ebola.head()

# Ifwewanttosetthisdaterangeastheindex,weneedtofirstsetthedateastheindex.
 ebola_5.index=ebola_5['Date']
# Nextwecan.reindex()ourdata.
 ebola_5 = ebola_5.reindex(head_range)
 print(ebola_5.iloc[:,:5])