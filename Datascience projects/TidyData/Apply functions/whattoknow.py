#Fundamental in Data cleaning 
#its like writing  a for loop or a map()  call to a function 

# used to perform calculations on the series and dataframes 

def my_function(): # define a new function called my_function
    # indentation for
    # function code
    pass # this statement is here to make a valid empty function

def my_sq(x):
 """Squares a given value """
     return x ** 2

def avg_2(x, y):
    """Calculates the average of 2 numbers """
    return (x + y) / 2


#using our function s
my_calc_1 = my_sq(4)
print(my_calc_1)

my_calc_2 = avg_2(10, 20)
print(my_calc_2)

#Applying the functions in pandas 
df = pd.DataFrame({"a":[10,20,30,], "b":[20,30,40]})
print(df)

affecting directly 

print(df['a'] ** 2)

#Apply method  to use our square method 
sq = df['a'].apply(my_sq)#we dont need to mpass the round brackets my_sq()
print(sq)

#if a function has two arguments the orthe is passed  as  keyword arfgument 


def my_exp(x,e):
    return x ** each

#using the function o
#in datascience 


ex = df['a'].apply(my_exp, e=2)
print(ex)

#WHEN WORKING WITH DATAFRAMES you specify either to apply on colums or rows using 
# column-wise, we can pass the axis=0 or axis="index"
# we want the function to work row-wise, we can pass the
# axis=1 or axis="columns" parameter into .apply().

#columnwise operation 
use axis = 0 parameter 

#Example 
 df=pd.DataFrame({"a": [10,20,30],"b":[20,30,40]})
 print(df)

df.apply(print_me, axis=0)

#using index , it only works when the col is passed to function 
#Example 
def avg_3(x, y, z):
    return (x + y + z) / 3


#applying it
def avg_3_apply(col):
 """The avg_3 function but apply compatible
 by taking in all the values as the first argument
 and parsing out the values within the function
 """
    x = col[0]
    y = col[1]
    z = col[2]
    return (x + y + z) / 3
print(df.apply(avg_3_apply))

#ROWISE OPERATION 
def avg_2_apply(row):
 """Takingtheaverage ofrowvalue.
 Assumingthatthere areonly2values inarow.
 """
    x=row[0]
    y=row[1]
    return (x+y)/2
print(df.apply(avg_2_apply,axis=0))


#Vectorise can be used to do the calculations easily 

import numpy as np
def avg_2_mod(x, y):
 """Calculate the average, unless x is 20
 If the value is 20, return a missing value
 """
    if (x == 20):
        return(np.NaN)
    else:
        return (x + y) / 2

#then 
print(avg_2_mod(10, 20))  #answer is 15
#with a vector it wont work for now 

 # will cause an error
 print(avg_2_mod(df['a'], df['b']))

#so to overcome we use NUMPY
#Vectorise with numba  and  numpy
# We want to change our function so that when it is given a vector of values, it will perform
#the calculations in an element-wise manner.
import numpy as np 
#np.vectorize actually  creates a new function
avg_2_mod_vec = np.vectorize(avg_2_mod)
#use the newly vectorized function 
print(avg_2_mod_vec)

#answer  [15. nan 35.]


#USING A DECORATOR
# This method works well if you do not have the source code for an existing function.
# However, if you are writing your own function, you can use a Python decorator to
# automatically vectorize the function without having to create a new function. A decorator
# is a function that takes another function as input, and modifies how that function’s
# output behaves.

#TO USE A VECTORIZE DECORATOR 
#We use the @ symbol befoore function definition 

@np.vectorize
def v_avg_mod(x,y):
    """Calculate the average, unless x is 20 Same as before,
     but we are using the vectorize decorator"""

    if (x == 20):
        return(np.NaN)
    else:
        return(x + y)/2

#we can directly use the vectorized function 
#without having to create a new function 
print(v_avg_mod(df['a'],df['b']))

#ans [15. nan 35.]

VECTORIZING WITH NUMBA 
# The numba library4 is designed to optimize Python code, especially calculations on arrays
# performing mathematical calculations. Just like numpy, it also has a vectorize decorator.

import numba 
@numba.vectorize
def v_avg_2_numba(x,y):
     """Calculatetheaverage, unlessxis20
         Usingthenumba decorator.
     """
    if (x == 20):
        return(np.NaN)
    else:
        return(x + y)/2

print(v_avg_2_numba(df['a'], df['b']))
# ValueError:Cannotdetermine Numbatypeof
 <class'pandas.core.series.Series'>
# We actually have to pass in the numpy array representation of our data using the.values
# attribute of our Series objects(ChapterR).

 #passingin thenumpyarray
 print(v_avg_2_numba(df['a'].values, df['b'].values))

#lambda functions 
#we use the lambda keyword 
df['a_sq_lamb']= df['a'].apply(lambda x:x**2)
 print(df)
