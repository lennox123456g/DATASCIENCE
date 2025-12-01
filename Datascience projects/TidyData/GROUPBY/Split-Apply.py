#split apply  and combine and groupby operations 
#They are a nice way to aggregate , trnsform and filter data

#separtar dat depending on keys 
#function called on each part of the data 
#results from each part are then  combined to create a new dataset


#some part of data might be used to make calculations 

#we use the pandas groupby
#but can be done without a groupby 

#example
#Aggregation can be done by using conditional subsetting on a dataframe
#Transformation can be done by passing a column into a separate function
#Filtering can be done with conditional subsetting

#However, when you work with your data using .groupby() statements, your code can
#be faster, you have greater flexibility when you want to create multiple groups, and you
#can more readily work with larger data sets on distributed or parallel systems.

#WHAT TO KNOW 
#understand what groupoe ddata is 
#calculate summaries of data  using .groupby operations 
#perform aggregation, transformation and filtering operations on groupoed data 
#separate data by groups for differnt calculations 


#AGGREGATION 
#to take multiple values and return a single value 
#example is  calculating mean 
#its like a summary of the data makes data smaller 

import pandas as pd 
df = pd.read_csv('data/gapminder.tsv',sep='\t')

#calculate the average life expectancy each year 
avg_life_exp_by_year = df.groupby('year')["lifeExp"].mean
print(avg_life_exp_by_year)

#getiing unique values 
#get a list of unique values by year 
years = df.year.unique()
print(years)


#subset the data for theyear 1952
y1952 = df.loc[df.year == 1952, :]
print(y1952)

#now agrregatintg that data 
y1952_mean = y1952["lifeExp"].mean


#deccrib efunction does qlot of statistics
# group by continent and describe each group
 continent_describe = df.groupby('continent')["lifeExp"].describe()
 print(continent_describe)


 # AGGREGATE FUNCTION OF NUMPY (pandas documentation says we should alias it as agg when using it)
 import numpy as np
 #calculatethe averagelifeexpectancyby continent
 #butuse thenp.meanfunction
 cont_le_agg=df.groupby('continent')["lifeExp"].agg(np.mean)
 print(cont_le_agg)
# Whenwepassinthefunctioninto.agg(),weonlyneedtheactual functionobject,wedo
# notneedto“call”thefunction.That’swhywewritenp.meanandnotnp.mean().Thisis
# similar to when we called.apply()

#CUSTOM FORMULA
#creating our own functions not in pandas or any librarry  #page 221
def my_mean(values):
    """My version of calculating a mean"""
    # get the total number of numbers for the denominator
    n = len(values)
    # start the sum at 0
    sum = 0
    for value in values:
    # add each value to the running sum
    sum += value
    # return the summed values divided by the number of values
    return sum / n


#we can now pass ur custm function in the agg method

 # use our custom function into agg
 agg_my_mean = df.groupby('year')["lifeExp"].agg(my_mean)
 print(agg_my_mean)


#CUSTOM functionn bbut with more parameters
#the other parameters to be used are passed as key word argumets 
def my_mean_diff(values, diff_value):
     """Differencebetweenthemean anddiff_value
    """
    n=len(values)
    sum=0
    for value invalues:
        sum+=value
    mean = sum/n
    return(mean - diff_value)


#calculatethe globalaveragelifeexpectancy mean
 global_mean=df["lifeExp"].mean()
 print(global_mean)

#customaggregation functionwithmultipleparameters
agg_mean_diff=(
     df .groupby("year")
     ["lifeExp"]
    .agg(my_mean_diff,diff_value=global_mean)#diif_value passed as a keyword argument 
    
 )
 print(agg_mean_diff)


 #uUSING MULTIPLE FUNCTIONS SIMULTANEOUSLY 
 #the are pased in a lsit 
  # calculate the count, mean, std of the lifeExp by continent
 gdf = (
    df
    .groupby("year")
    ["lifeExp"]
    .agg([np.count_nonzero, np.mean, np.std])
 )
 print(gdf)

 #ONE CAN USE A DICTIONARY 
 #ON A DATFRAME 
#  When specifying a dict on a grouped DataFrame, the keys are the columns of the
# DataFrame, and the values are the functions used in the aggregated calculation. This
# approach allows you to group one or more variables and use a different aggregation
# function on different columns simultaneously.


 # use a dictionary on a dataframe to agg different columns
 # for each year, calculate the
 # average lifeExp, median pop, and median gdpPercap
 gdf_dict = df.groupby("year").agg(
 {
    "lifeExp":"mean",
    "pop":"median",
    "gdpPercap":"median"
 }
 )
 print(gdf_dict)


#ON SERIES 
#here colums a renamed afyter the operation
 gdf=(
    df
    .groupby("year")["lifeExp"]
    .agg([
        np.count_nonzero,
        np.mean,
        np.std,
    ]
 )
 .rename(
    columns={
        "count_nonzero":"count",
        "mean":"avg",
        "std":"std_dev",
 }
 )
 .reset_index() # return a flat dataframe
 )
 print(gdf)


 #TRANSFORM 184
 transform takes values and  just transforms them.
 unlike agg() that returns a single aggreagated value

 #z score example
 xis a data point in our data set
 µis the average of our data set, as calculated by Equation 8.1
 σ is the standard deviation, as calculated by Equation 8.3


 #python function that calculates z-score 
 def my_zscore(x):
    ''' calculates the z-score of the provided data
    'x' is a vector os series of values '''

    return ((x-x.mean()) /x.std())

    #THis is a custom z_score

#Now we can use this function to .transform() our data by group.

#zcore by SCIPY LIBRARY
from scipy.stats import zscore
#calculate the a grouped zscore
sp_z_grouped = df.groupby('year')["lifeExp"].transform(zscore)




# calculate a nongrouped zscore
 sp_z_nogroup = zscore(df["lifeExp"])

  # grouped z-score
 print(transform_z.head())

 # grouped z-score using scipy
 print(sp_z_grouped.head())

  # nongrouped z-score
 print(sp_z_nogroup[:5])

# Our grouped results are similar. However, when we calculate the z-score outside the
# .groupby(), we get the z-score calculated on the entire data set, not broken out by group



#MISSING VALUES 
we use the interpolate() method or forward/backward filling our data.
#we could fill with the mean of the column 
#or fill based on a particular group
#.fillna()methodtofill in the missing

#Example working with tips dataset

import seaborn as sns 
import numpy as np

#set the seEd so that results are deterministic
np.random.seed(42)

#sample 10 rows from tipos 
tips_10 = sns.load_dataset("tips").sample(10)

#randomly pick 4 'total_bill' values and return them into missing tips
tips_10 [
    np.random.permutation(tips_10.index)[:4],
    "total_bill"
] = np.NaN

print(tips_10)

# We can usethe.groupby() method to calculate a statistict of ill in missing values.
# Instead of using.agg(),weusethe.transform()method.First, let’s count the
# non-missingvaluesbysex.


count_sex = tips_10.groupby('sex').count()
print(count_sex)

#result gives as the missing value for each of the sexes in each colom

#noew lets calcilate the grouped average and use it to fill the missing values 

def fill_na_mean(x):
    """ returns the average of a given vector"""
    avg = x.mean()
    return x.fillna(avg)


    #calculate the mean 'total bil by sex'
    total_bill_group_mean = (
        tips_10
        .groupby("sex")
        .total_bill
        .transform(fill_na_mean)
)

#assign to a new column in the original data 
#you can also replcae the original column with total_bill
tips_10["fill_total_bill"] = total_bill_group_mean


# If we just look at the two total_bill columns, we see that different values were filled
# in for the NaN missing values.
print(tips_10[['sex', 'total_bill', 'fill_total_bill']])


#FILTER

#last action one can do with groupby 
#alllows you to split data by keys  an dthen perform a king of boolean substeetting on the data
# we may want to work with may me data with more than 30 observation s, so we use filter 


#load the tips dataset 
tips = sns.load_dataset('tips')

#note the number of rows in the original data 
print(tips.shape)


#look at the frequesncy counts for the table size 
print(tips['size'].value_counts())


# The output shows that table sizes of 1, 5, and 6 are infrequent. Depending on your
# needs, you may want to filter those data points out. In this example, we want each group
# to consist of 30 or more observations.
# To accomplish this goal, we can use the .filter() method on a grouped operation.

#filter the dat such that each group has more than 30 observations 
tips_filtered =(
    tips.groupby("size")
    .filter(lambda x: x["size"].count >= 30)
)

#the out put shows filtered out data

print(tips_filtered['size'].value_counts())


#CONSOLIDATING GROUPS 
#certain times we want ofirst store te grouoed data then later do calculation of agg, transfoform and filter later 

tips_10 = sns.load_dataset('tips').sample(10, random_state=42)
print(tips_10)

# Wecanchoosetosavejustthegroupbyobjectwithoutrunninganyother.agg(),
# .transform(),or.filter()methodonit.

#save just the grouped object 
grouped = tips_10.groupby('sex')

#note that we just get back the object and its memroy location
print(grouped)

#to check the groups use 
 # see the actual groups of the groupby
 # it returns only the index
 print(grouped.groups)

#This approach does allow you to save just the grouped result. You could then perform
# multiple .agg(), .transform(), or .filter() operations without having to process the
# .groupby() statement again.



#group calculations of multiple variables 
#calculating mean of relevat columns 
avgs = grouped.mean()
print(avgs)


#here not ;all columns report a mean , those that dont have numericals are left out 

#try listing all columns to see 
print(tips_10.columns)

#SELECTING A GROUPP 

#get a female group 
female = grouped.get_group('Female')
print(female)

#Iterating throught the groups 

#we can write functions 
for sex_group in grouped:
    print(sex_group)


for sex_group in grouped:
    # get the type of the object (tuple)
    print(f'the type is: {type(sex_group)}\n')
    # get the length of the object (2 elements)
    print(f'the length is: {len(sex_group)}\n')
    # get the first element
    first_element = sex_group[0]
    print(f'the first element is: {first_element}\n')
    # the type of the first element (string)
    print(f'it has a type of: {type(sex_group[0])}\n')
    # get the second element
    second_element = sex_group[1]
    print(f'the second element is:\n{second_element}\n')
    # get the type of the second element (dataframe)
    print(f'it has a type of: {type(second_element)}\n')
    # print what we have
    print(f'what we have:')
    print(sex_group)
    # stop after first iteration
    break



# We have a two-element tuple in which the first element is a str (string) that represents
# the Male key, and the second element is a DataFrame of the Male data.
# If you prefer, you can forgo all the techniques introduced in this chapter and iterate
# through your grouped values in this manner to perform your calculations. Again, there
# may be times when this is the only way to get something done. Perhaps you have a
# complicated condition you want to check for each group, or you want to write out each
# group into separate files. This option is available to you if you need to iterate through the
# groups one at a time.

#note 
#youcan't reallygetthe0 elementfromthegroupedobject
 print(grouped[0])



#WORKING ON MULTIPLE VARIBLES WE USE A LIST 
#MEAN by sex and time 
bill_sex_time = tips_10.groupby(['sex', 'time'])
group_avg = bill_sex_time.mean()


#FLATTENING  THE RESULTS USING RESTE INDEX 
#so the  feedback we get when we check 
#type of the grouped data is weird 
print(type(group_avg))

#the results look strange ,they appear to be empty cells in a dataframe 

#if we look at columns we get what we expect
print(group_avg.columns)

#however more interesting things happen when we look at index 
print(group_avg.index)

#so solve this , we can use MULTIINDEX .if we want we can get a regular flat dataframe 
#back we can camll the reset_index() method

group_method = tips_10.groupby(['sex','time']).mean().reset_index()
print(group_method)

#Alternatively one can use index=False parameter 
 group_param = tips_10.groupby(['sex', 'time'], as_index=False).mean()
 print(group_param)


#WORKING WITH MULTIINDEX 
#Sometimes, you may want to chain calculations after a .groupby() method. You can
# always “flatten” the results and then execute another .groupby() statement, but that may
# not always be the most efficient way of performing the calculation.
# We begin with epidemiological simulation data on influenza cases in Chicago (this is a
# fairly large data set).

#Notice that we can even read a compressed zip file of a csv 
intv_df = pd.read_csv('data/epi_sim.zip')
print(intv_df)


#EXAMPLE
count_only = (
 intv_df
 .groupby(["rep", "intervened", "tr"])
 ["ig_type"]
 .count()
 )
 print(count_only)

#  Now that we’ve done a .groupby() .count(), we can perform an additional
# .groupby() that calculates the average value. However, our initial .groupby() method
# does not return a regular flat dataframe.
 print(type(count_only))

#  Instead, the results take the form of a multi-index series. If we want to do another
# .groupby() operation, we have to pass in the levels parameter to refer to the multi-index
# levels. Here we pass in [0, 1, 2] for the first, second, and third index levels, respectively

 count_mean = count_only.groupby(level=[0, 1, 2]).mean()
 print(count_mean.head())

# We can combine all of these operations in a single command.
count_mean  =(
  intv_df
 .groupby rep
 .count
 .groupbylevel
 .mean()
)


import seaborn as sns 
import matplotlib as plt 
fig = sns.implot(
    data = count_mean.reset_index(),
    x = "intervened",
    y = "ig_type",
    hue = "rep",
    col = "tr",
    fit_reg = False
    palette = "viridis"
)

# The previous example showed how we can pass in a level to perform an additional
# calculation. It used integer positions, but we can also pass in the string of the
# level to make our code a bit more readable.
# Here, instead of looking at the .mean , we will be using .cumsum for the cumulative
# sum.

cumulative_count = (
    intv_df
    .groupby(["rep","intervened", "tr"])
    ["ig_type"]
    .count()
    .groupby(level=["rep"])
    .cumsum()
    .reset_index
)

fig sns.implot(
    data = cumulative_count,
    x = "intervened",
    y = "ig_type",
    hue = "rep",
    col = "tr",
    fit_reg = False
    palette = "viridis"

)