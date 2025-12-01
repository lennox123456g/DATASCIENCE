#this is the chapter that gets one into machine learning 

# Andreas Müller and Sarah Guido’s Introduction to Machine Learning
 with Python
# . Sebastian Raschka and Vahid Mirjalili’s Python Machine Learning
# . Mark Fenner’s Machine Learning with Python for Everyone
# . Andrew Kelleher and Adam Kelleher’s Machine Learning in Production:
# Developing and Optimizing Data Science Workflows and Applications



#most of the work before was to understand what our coklumns hold and organising them in particulr 
#the variable we are trying to predict 

#if our outcome variable is continuous , we  use a linear refgression model
#if it is binary , we use logistic egression 
#if  if it is count data , we use  a Poisson model 
#wehn looking for an outcome of interest but also have censoring , we use Survival models 
# We compare model diagonistices when  we fitting models for prediction sice we need to find a way to choose the best model 
#Regularlissation is used hen  we are solely interested in prediction rther than infereence , we do this to make the model more stable 
# If we do not have an outcome variable we can test our model against, we
# would use some kind of unsupervised modeling technique, such as clustering


#LINEAR REGRESSION
#Its role is to draw a straightline relationship between 
#a response (out come or dependable variable) and a predictor variable(independent  variable , covariate,etc )

#example 
import  pandas as pd 
import seaborn as sns 

tips = sns.load_dataset('tips')
print(tips)


#so we want  to see how total bill  predicts the tip 


#We shall use STATSMODELS LIBRARY 
import statsmodels.formula.api as smf  
#to perform this linear regression , we use the ols( functuion)

#which computes the  ordinary  least squares value; it is one method lto lestimate lparametersinal linear
# regression. Recall that the formula for a line is y=mx+b, where y is our response
# variable,x is our predictor,b is the intercept,and m is the slope, the parameter we
# are estimating.
# The formula notation has two parts,separated by a tilde,~.To the left of the tilde is the
# response variable,and to the right of the tilde are the predictor(s).

model = smf.ols(formula='tip ~ total_bill', data = tips)
#In datscience is a mathematical way to describe how things relate to each other 

#once we have specified  our model,we can fit the data to the model by using the fit method 
results = model.fit()

#to look at results , we call the summary method () on the results 
print(results.summary())

#y = (0.105)x +0.920.  check  the results 
#To interpret these numbers,
# we say: for every one unit increase in total_bill (i.e., every time the bill increases by a
# dollar), the tip increases by 0.105 (i.e., 10.5 cents).


#if we just want coeefecients we use params
print(results.params)


#A comfidene interval is also vital 
#Depending on your field, you may also need to report a confidence interval, which
# identifies the possible values the estimated value can take on. The confidence interval
# includes the values less than [0.025
# .conf_int() method.

print(results.conf_int)


#performing this analysis rather with Scikit learn 
#SCIKIT LEARN 
#we import the linear_model module from this library 
from sklearn import linear_model

#we now create own linear regression object 

#Next, we need to specify the predictor, X, and the response, y. To do this, we pass in the

lr = linear_model.LinearRegression()
# columns we want to use for the model.


#note that the parameters capital X and lower y .This comes from a 
# a mathematical notation, where predictors,X are a matrix of  values and response y, is a vector of values 


#Example 
#X doesnt take a single parameter so an error will be returned 
#Example 
 # note it is an uppercase X
 # and a lowercase y
 # this will fail because our X has only 1 variable
 predicted  = lr.fit(X=tips['total_bill'], y = tips['tip'])

# Reshape your data either using array.reshape(-1, 1) if your data has a
# single feature or array.reshape(1,-1) if it contains a single sample.

#Depending on whether we have a single feature (which is the case
# here) or a single sample (i.e., multiple observations), we will specify reshape(-1, 1) or
# reshape(1,-1), respectively.

#calling reshape on the clumn causes and error 
#this will fail 
predicted = lr.fit(
    X = tips["total_bill"].reshape(-1,1), y =tips["tip"]

    #to properly reshape the data, we must use values  attribute 

    #whe we call the .values on a pandas dataframe we get the numpy representation of the data 
)

#Since sklearn is built to take numpy arrays, there will be times when you have to do
# some data manipulations to pass your dataframe into sklearn.

#To obtain the coefficients in
# sklearn, we call the .coef_ attribute on the fitted model.

print(predicted.coef)


#to get the intercept 
print(pridicted.intercept_)

#MULTIPLE REGRESSION 
# In simple linear regression, one predictor is regressed on a single response variable.
# Alternatively, we can use multiple regression to put multiple predictors in a model.#


#WITH STATS MODELS 
#Example 
#niote the fit() chain method at the end 
model = smf.ols(formula="tip  ~ total_bill + size", data=tips).fit()

print(model.summary())

# The interpretations are exactly the same as before,except each parameter is interpreted
# “with all other variable sheld constant.”That is, for every one unit increase(dollar) in
# total_bill,the tip increases by 0.09(i.e.,9cents)as long as the size of the group does not
# change.

#WITH SCIKIT LEARN 
lr = linear_model.LinearRegression()

#since we are performing multiple regression
#we nolonger need to reshape our X values 
predicted = lr.fit(X=tips[["total_bill", "size"]], y=tips["tip"])
print(predicted.coef_)
# we can also get the intercept
print(predicted.intercept_)

#CATEGORICAL VARIABLES WITH STATS MODELS 
#statsmodels will automatically create dummy variables for us. To avoid multicollinearity,
# we typically drop one of the dummy variables. That is, if we have a column that indicates
# whether an individual is female, then we know if the person is not female (in our data),
# that person must be male. In such a case, we can effectively drop the dummy variable that
# codes for males and still have the same information.

#Exmaple that uses all our variables in  our data 
model = smf.ols(
    formula = "tip ~ total_bill + size + sex + smoker + day + time"
    data = tips
).fit()

#We can see from the summary that statsmodels automatically creates dummy variables
# aswell as drops the reference variable to avoid multicollinearity.
 print(model.summary())


# The interpretation of the continuous(i.e.,numeric) parameters is the same as before.
# However, our   interpretation  of  categorical variables must be stated in relation to the
# reference variable(i.e., thedummy variable that was dropped from the analysis).For
# example,the coefficient for sex[T.Female] is 0.0324. We interpret this value in relation to
# the reference value,Male; that is,we say that when the sex of the server “changes”  from
# Male to Female, the tip increases by 0.324.For the day variable:

print(tips.day.unique())


#We see that our.summary() is missing   Thur,so that is the reference variable to use to
# interpret the coefficients.

#CATEGORICAL DATA WITH SCIKIT LEARN 
#here we manually create  the dummy variables 
#we can use the pandas get_dummies
#This  function converts all the
# categorical variables into dummy variables automatically,so we do not need to pass in
# individual columns one atatime. sklearn has a One Hot Encoder function that does
# somethingsimilar


#DUMMY VARIABLES IN PANDAS 
tips_dummy = pd.get_dummies(
    tips[[total_bill,"size","sex","smoker","day","time"]]
)

print(tips_dummy)

#to drop the reference varuable  we use drop_first= True
x_tips_dummy_ref = pd.get_dummies(
    tips[[total_bill,"size","sex","smoker","day","time"]],
    drop_first=True
)
print(x_tips_dummy_ref)

#now lets fit the model 
lr= linear_model(X=x_tips_dummy_ref, y=tips["tip"])

#lets obtain the coeffecients  and intercept
print(predicted.intercept_)
 print(predicted.coef_)

#KEEPING INDEX LABELS FROM SKLEARN 
import numpy as np

#cretae and fit the model 
lr = linear_model.LinearRegression()
predicted = lr.fit(X=x_tips_dummy_ref, y=tips["tip"])

#get the intercept a along with other coeffecients 
values = np.append(predicted.intercept_, predicted.coef_)

#get the names of the values 
names = np.append("intercept", x_tips_dummy_ref.columns )

# put everything in a labeled dataframe
results = pd.DataFrame({"variable": names, "coef": values})

print(results)


#ONE HOT ENCODING IN SCIKT LEARN WITH TRANSFROMER PIPELINES 
# Scikit-learn has its own way of processing data for analysis using “pipelines.” We can use
# the one-hot encoding transformer in a pipeline to process our data in scikit-learn, instead
# of pandas, before we fit our model.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import pipeline

# We first need to specify which columns we want to process, here we are only looking
# to work with categorical variables.

categorical_features = ["sex", "smoker", "day", "time"]
categorical_transformer = OneHotEncoder(drop="first")

#Once we have the columns and the processing step we want, we can then pass the steps
# into ColumnTransformer(). Since we want to still have the numeric variables in the final
# model, but didn’t specify a processing step for them, we pass in remainder="passthrough"
# to make sure those variables not specified in the transformers step still make it to the
# final model.

 preprocessor = ColumnTransformer(
    transformers=[
    ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough", # keep the numeric columns
 )

#  Finally, we can create a Pipeline() with all the preprocessing steps, and then to the
# model we want.
 pipe = Pipeline(
    steps=[
    ("preprocessor", preprocessor),
    ("lr", linear_model.LinearRegression()),
    ]
 )



# Finally, we can fit our model just like before.
    pipe.fit(
    X=tips[["total_bill", "size", "sex", "smoker", "day", "time"]],
    y=tips["tip"],
 )

Pipeline(steps=[('preprocessor',
    ColumnTransformer(remainder='passthrough',
    transformers=[('cat',
    OneHotEncoder(drop='first'),
    ['sex','smoker','day',
    'time'])])),
    ('lr',LinearRegression())]
 )

 print(type(pipe))
# Wecan’tgetthe.intercept_andcoef_becausethePipeline(), isnota
# LinearRegression()object.

# We need to access the coefficients  in an additional step.This is because not all models
# will haveintercept_and coef_values, the Pipeline()is a generic function that works
# with any model with in the sklearn library

 #combinethe interceptandcoefficientsinto singlevector
 coefficients=np.append(
 pipe.named_steps["lr"].intercept_, pipe.named_steps["lr"].coef_
 )
 #combinethe intercepttextwiththe otherfeaturenames
 labels=np.append(
 ["intercept"],pipe[:-1].get_feature_names_out()
 )
 
 #createa dataframeofallthe results
 coefs=pd.DataFrame({"variable":labels,"coef":coefficients})
 print(coefs)