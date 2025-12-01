#for comparing  models  ad measuring their perfomance 

#RESIDUALS 
#The residuals of a  model compare what a model calculates and the actual values in the dat.
#Lets fit some models on a housing dataset 

import pandas as pd 
housing = pd.read_csv('data/housing_renamed.csv')

print(housing.head())

#we will begin with a multiple linear regression model with three covarieties 
import statsmodels 
import statsmodels.api as sm 
import statsmodels.formula.api as smf
house_1= smf.glm(
    "value_per_sq_ft ~ sq_ft + boro",
    data=housing
).fit()

print(house.summary())

#We can plot the residuals of ourmodel (Figure16.1).Whatwearelookingforisaplot
# with a random scattering of points.If a pattern is apparent, then we will need to
# investigate our data and model to see why this pattern emerged.

import seaborn as sns 
import matplotlib.pyplot as plt 

fig ax=plt.subplots()
sns.scatterplot(
    x=house1.fittedvalues, y=house1.resid_deviance, ax = ax 
)

plt.show()

# This residual plot is concerning because it contains obvious clusters and groups(residual
# plots are supposed to look random).We can color our plot by the boro variable,which
# indicates the borough of NewYork where the data apply(Figure16.2).


#get the data used for the residual plot and boro color 
res_df = pd.DataFrame(
    {
        "fittedvalues":  house.fittedvalues #get model attributes 
        "resid_deviance": house1.resid_deviance,
        "boro":housing["boro"], #get a value from the data column
    }
)
 # greyscale friendly color palette
 color_dict = dict(
 {
    "Manhattan": "#d7191c",
    "Brooklyn": "#fdae61",
    "Queens": "#ffffbf",
    "Bronx": "#abdda4",
    "Staten Island": "#2b83ba",
 }
 )
 fig, ax = plt.subplots()
 fig = sns.scatterplot(
    x="fittedvalues",
    y="resid_deviance",
    data=res_df,
    hue="boro",
    ax=ax,
    palette=color_dict,
    edgecolor='black',
 )
 plt.show()

#  When we color our points based on boro, you can see that the clusters are highly
# governed by the value of this variable.


#Q0Q PLOT 
# A q-q plot is a graphical technique that determines whether your data conforms to a
# reference distribution. Since many models assume the data is normally distributed, a q-q
# plot is one way to make sure your data really is normal (Figure 16.3).

from scipy import stats 


#make a copy of the variable so we dont keep typing it
resid = house.resid_deviance.copy()

fig = statsmodels.graphics.gofplots.qqplot(resid, line='r')
plt.show()


#we can also plot a histogram of residuals to see if our data is normal 
residual_std = stats.zscore(resid)

fig, ax = plot.subplots()
sns.histplot(resid_std, ax = ax)
plt.show()

# If the points on the q-q plot lie on the red line, that means our data match our
# reference distribution. If the points do not lie on this line, then one thing we can do is
# apply a transformation to our data. Table 16.1



#COMPARING MULTIPLE MODELS 
#we compare them so that we can chooose the best one 

# We begin by fitting five models. Note that some of the models use the + operator to add
# covariates to the model, whereas others use the * operator. To specify an interaction in
# our model, we use the * operator. That is, the variables that are interacting are behaving in
# a way that is not independent of one another, but in such a way that their values affect one
# another and are not simply additive.

#nOTE 
# If the original housing data set had a column named "class", this would cause an error
# because "class" is a Python keyword. Therefore, the column was renamed "type".

f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ! units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft * boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'

house1 = smf.ols(f1, data=housing).fit()
house2 = smf.ols(f2, data=housing).fit()
house3 = smf.ols(f3, data=housing).fit()
house4 = smf.ols(f4, data=housing).fit()
house5 = smf.ols(f5, data=housing).fit()

#With all our models, we can collect all of our coefficients and the model with which
# they are associated

mod_results = (
    pd.concat(
        [
            house1.params,
            house2.params,
            house3.params,
            house4.params,
            house5.params,
        ],
        axis=1,
    )
    .rename(columns=lambda x:"house" + str(x + 1))
    .reset_index()
    .rename(columns={"index": "param"})
    .melt(id_vars="param", var_name="model", value_name="etsimate")
)

print(mod_results)

# Since it’s not very useful to look at a large column of values, we can plot our
# coefficients to quickly see how the models are estimating parameters in relation to each
# other 

color_dict = dict(
    {
        "house1":  "#d7191c",
        "house2": "#fdae61",
        "house3": "#ffffbf",
        "house4": "#abdda4",
        "house5": "#2b83ba",
    }
)



#the figure 
fig, ax = plt,subplots()
ax = sns.pointplot(
    x="estimate",
    y="param",
    hue="model",
    data=mod_results,
    dodge = True, #jitter the points 
    join=False #dont join the points 
    palette=color_dict
)

plt.tight_layout()
plt.show()

#now that we have our linear moeels 
#we can use the the analysis of Variance  method to compare them 
#The ANOVA will give us the residual sum of squares (RSS),
# which is one way we can measure performance (lower is better).


model_names = ["house1","house2", "house3","house4", "house5"]
house_anova = satsmodels.stats.anova.anova_lm(
    house1, house2,house3,house4,house5
)
house_anova.index = model_names

print(house_anova)


#CALCULATING MODEL PERFOMANCE USING AKAIKE INFORMATION  CRITERION  AND 
#BAYES INFORMATION  CRITERION
#These methodsa pply  a
# penalty for each feature that isa dded to the model (lowerAIC and BIC value is better).
# Thus,we should strive to balance performance and parsimony.

house_models = [house1, house2,house3, house4,house5]

abic = pd.DataFrame{
    "model": model_names,
    "aic": [mod.aic for mod in house_models],
    "bic": [mod.bic for mod in house_models],
}

print(abic.sort_value(by=["aic","bic"]))


#WORKING ON GENERALISED LINEAR MODELS
#We can perform the same calculations and model diagnostics on generalized linear models
# (GLMs).We can use the deviance of the model todomodelcomparisons:

def deviance_table(*models):
    """ create a table of model diagonistics from model objects """

    return pd.DataFrame(
        {
            "df_residuals": [mod.df_resid for mod in models],
            "resid_stddev": [mod.df_model for mod in models],
            "deviance": [mod.deviance for mod in models],
        }
    )

f1 = 'value_per_sq_ft ~ units + sq_ft + boro'
f2 = 'value_per_sq_ft ~ units * sq_ft + boro'
f3 = 'value_per_sq_ft ~ units + sq_ft * boro + type'
f4 = 'value_per_sq_ft ~ units + sq_ft * boro + sq_ft * type'
f5 = 'value_per_sq_ft ~ boro + type'

glm1 = smf.glm(f1, data=housing).fit()
glm2 = smf.glm(f2, data=housing).fit()
glm3 = smf.glm(f3, data=housing).fit()
glm4 = smf.glm(f4, data=housing).fit()
glm5 = smf.glm(f5, data=housing).fit()

glm_anova = deviance_table(glm1, glm2,glm3,glm4,glm5)
print(glm_anova)

#NOW DOING THE SAME KIND OF CALCULATION ON  A LOGISTIC REGRESSION 
#create a binary variable 
housing["high"] = (housing["value_per_sq_ft"] >= 150.astype(int))
print(housing["high"].value_counts())


 # create and fit our logistic regression using GLM

f1 = "high ~ units + sq_ft + boro"
f2 = "high ~ units * sq_ft + boro"
f3 = "high ~ units + sq_ft * boro + type"
f4 = "high ~ units + sq_ft * boro + sq_ft * type"
f5 = "high ~ boro + type"

logistic = statsmodels.genmod.families.family.Binomial(
    link=statsmodels.genmod.families.links.Logit()
)

glm1=smf.glm(f1,data=housing,family=logistic).fit()
glm2=smf.glm(f2,data=housing,family=logistic).fit()
glm3=smf.glm(f3,data=housing,family=logistic).fit()
glm4=smf.glm(f4,data=housing,family=logistic).fit()
glm5=smf.glm(f5,data=housing,family=logistic).fit()


 #show the deviances from our GLM models
 print(deviance_table(glm1,glm2,glm3, glm4,glm5))


#Finally,we can create a table of AIC and BIC values.
mods=[glm1,glm2,glm3, glm4,glm5]

abic_glm = pd.DataFrame(
    {
        "model": model_names,
        "aic":[mod.aic for mod in house_models],
        "bic":[mod.bic formodin house_models],
    }
)

print(abic_glm.sort_values(by=["aic","bic"]))

#Looking at all these measures,we can say Model4 is performing the best sofar. 


#COMPRING MODELS USING K-FOLD CROSS VALIDATION 
# Cross-validationisanothertechniquetocomparemodels.Oneofthemainbenefitsisthat
# itcanaccountforhowwellyourmodelperformsonnewdata.Itdoesthisbypartitioning
# yourdataintokparts.Itholdsoneofthepartsasideasthe“test”setandthenfitsthe
# model on the remaining k − 1 parts, the “training” set. The fitted model is then used on
# the “test” and an error rate is calculated. This process is repeated until all k parts have been
# used as a “test” set. The final error of the model is some average across all the models.
# Cross-validation can be performed in many different ways. The method just described
# is called “k-fold cross-validation.” Alternative ways of performing cross-validation include
# “leave-one-out cross-validation,” in which the training data consists of all the data except
# one observation designated as the test set.
# Here we will split our data into k − 1 testing and training data sets

from sklearn.model.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

print(housing.columns)

#get training and test data 
X_train, X_test, y_train, y_test = train_test_split(
    pd.get_dummies(
        housing[["units", "sq_ft","boro"]], drop_first=True
    ),
    housing["value_per_sq_ft"],
    test_size=0.20,
    random_state = 42,
)

# Danger
# Pay attention to the capitalization of the letter X when looking at scikit-learn tutorials and
# documentation. This is a convention that comes from matrix notation from statistics and
# mathematics.

#We get the score of how well our model is performing using our test data 
lr = LinearRegression().fit(X_train,y_train)
print(lr.score(X_text,y_test))

# Since sklearn relies heavily on the numpy ndarray, the patsy library allows you to
# specify a formula just like the formula API in statsmodels, and it returns a proper numpy
# array you can use in sklearn.

# Here is the same code as before, but using the dmatrices function in the patsy library.
from patsy import dmatrices 

y, X = dmatrices (
    "value_per_sq_ft ~ units + sq_ft + boro",
    housing,
    return_type="dataframe",
)
X_train, X_test,y_train, y_test = train_test_split(
    X,y, test_size=0.20, random_state=42
)

lr= LinearRegression().fit(X_train, y_train)
print(lr.score(X_test,y_test))

#To perform a k-fold cross-validation, we need to import this function from sklearn.
from sklearn.model import KFold, cross_val_score

#get a fresh new housing dataset 
housing = pd.read_csv('data/housing_renamed.csv')

# We now have to specify how many folds we want. This number depends on how many
# rows of data you have. If your data does not include too many observations, you may opt
# to select a smaller k (e.g., 2). Otherwise, a k between 5 to 10 is fairly common. However,
# keep in mind that the trade-off with higher k values is more computation time.

kf = KFold(n_splits=5)
y, X = dmatrices('value_per_sq_ft ~ units + sq_ft + boro', housing)

#now we can train and test data on each fold 
cofs = []
scores = []
for train, test in kf.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    lr = LinearRegression().fit(X_train,y_train)
    coefs.append(pd.DataFrame(lr.coef))
    scores.append(lr.score(X_test, y_test))


#we can also view the results 
coefs_df = pd.concat(coefs)
coefs_df.columns = X.design_info.column_names
print(coefs_df)

# We can take a look at the average coefficient across all folds using .apply() and the
# np.mean() function.
import numpy as np 
print(coefs_df.apply(np.mean))

# We can also look at our scores. Each model has a default scoring method.
# LinearRegression(), for example, uses the R2 (coefficient of determination) regression
# score function.1
 print(scores)

 #We can also use cross_val_scores (for cross-validation scores) to calculate our scores.
 #use cross_val-scores to calculate CV scores 
 model = LinearRegression()
 scores = cross_val_score(model, X, y, cv=5)
 print(scores)

# When we compare multiple models to one another, we compare the average of the
# scores.

print(scores.mean())


# Now we’ll refit all our models using k-fold cross-validation.

#create the predictor and response matrices 
y1, X1 = dmatrices(
    "value_per_sq_ft ~ units + boro",housing)
y2, X2 = dmatrices ('value_per_sq_ft ~ units * boror + type', housing)
y3, X3 = dmatrices(
    "value_per_sq_ft ~ units + sq_ft*type", housing
)
y4, X4 = dmatrices(
    "value_per_sq_ft ~ units + aq_ft*boro + sq_ft*type", housing
)
 y5, X5 = dmatrices("value_per_sq_ft ~ boro + type", housing)

  # fit our models
 model = LinearRegression()
 scores1 = cross_val_score(model, X1, y1, cv=5)
 scores2 = cross_val_score(model, X2, y2, cv=5)
 scores3 = cross_val_score(model, X3, y3, cv=5)
 scores4 = cross_val_score(model, X4, y4, cv=5)
 scores5 = cross_val_score(model, X5, y5, cv=5)


#We can now look at our cross-validation scores.
 scores_df = pd.DataFrame(
 [scores1, scores2, scores3, scores4, scores5]
 )
 print(scores_df.apply(np.mean, axis=1))

#Once again, we see that Model 4 has the best performance.

# When we are working with models, it’s important to measure their performance.
# Using ANOVA for linear models, looking at deviance for GLM models, and using
# cross-validation are all ways we can measure error and performance when trying to
# pick the best model