#In Chapter 16, we considered various ways to measure model performance. Section 16.3
# described k-fold cross-validation, a technique that tries to measure model performance by
# looking at how it predicts on test data. This chapter explores regularization, one technique
# to improve performance on test data

#WHY REGULARISE 
#lets begin with a base case of liner regerssion 
#we will be using ACS data 

import pandas as pdacs = pd,read_csv('data/acs_ny.csv')
print(acs.columns)

# Now, let’s create our design matrices using patsy.

from patsy import dmatrices 

#sequential strings get concatenated togeter in python 
response, predictors = dmatrices(
    "FamilyIncome ~ NumBedrooms + NumChildren + NumPeople +"
    "NumRooms + NumUnits + NumVehicles + OwnRent +"
    "YearBuilt + ElectricBill + FoodStamp + HeatingFuel +"
    "Insurance + Language"
    data= acs,
)

#With our predictor and response matrices created, we can use sklearn to split our data
# into training and testing sets.

from sklearn.model_selction import train_test_split 

X_train, X_test,y_train, y_test = train_test_split(
    predictors,response, random_state=0
)

# Now, let’s fit our linear model. Here we are normalizing our data so we can compare
# our coefficients when we use our regularization techniques

from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 

lr= make_pipeline(
    StandardScaler(with_mean=False), LinearRegression()
)

lr= lr.fit(X_train,y_train)
print(lr)


Pipeline(steps=[('standardscaler',StandardScaler(with_mean=False)),
('linearregression', LinearRegression())])

model_coefs = pd.DataFrame(
    data=list(
        zip(
            predictors.design_info.column_names,
            lr.named_steps["linearregression"].coef_[0]
        )
    ),
    columns=["variable", "coef_lr"],
)

print(model_coefs)


#Now we can look at our model scores 
#score on training data 
print(lr.score(X_train,y_train))

#score on the testing data 
print(lr.score(X_test, y_test))

#In this particular case, our model demonstrates poor performance. In another potential
# scenario, we might have a high training score and a low test score—a sign of overfitting.
# Regularization solves this overfitting issue, by putting constraints on the coefficients and
# variables. This causes the coefficients of our data to be smaller. In the case of LASSO (least
# absolute shrinkage and selection operator) regression, some coefficients can actually be
# dropped (i.e., become 0), whereas in ridge regression, coefficients will approach 0, but are
# never dropped.

#LASSO REGRESSION
# The first type of regularization technique is called LASSO, which stands for least absolute
# shrinkage and selection operator. It is also known as regression with L1 regularization

#WE will fit the same model the way we did in our linear regression 

from sklearn.linear_model import LASSO
lasso= make_pipeline(
    StandardScaler(with_mean=False),
    Lasso(max_iter=1000, random_state=42),
)

lasso= lasso.fit(X_test, y_test)
print(lasso)

Pipeline(steps=[('standardscaler', StandardScaler(with_mean=False)),
 ('lasso', Lasso(max_iter=10000, random_state=42))])

#Now, let’s get a dataframe of coefficients, and combine them with our linear regression
# results.
 coefs_lasso = pd.DataFrame(
 data=list(
    zip(
        predictors.design_info.column_names,
        lasso.named_steps["lasso"].coef_.tolist(),
        )
    ),
    columns=["variable", "coef_lasso"],
 )
 model_coefs = pd.merge(model_coefs, coefs_lasso, on='variable')
 print(model_coefs)

#  Notice that the coefficients are now smaller than their original linear regression values.
# Additionally, some of the coefficients are now 0.
# Finally, let’s look at our training and test data scores.

print(lasso.score(X_train, y_train))

print(lasso.score(X_test, y_test))

#There isn’t much difference here, but you can see that the test results are now better
# than the training results. That is, there is an improvement in prediction when using new,
# unseen data.

#RIDGE REGRESSION Now let’s look at another regularization technique, ridge regression. It is also known as

# regression with L2 regularization.
# Most of the code will be very similar to that seen with the previous methods. We will
# fit the model on our training data, and combine the results with our ongoing dataframe
# of results.

from sklearn.linear_model import Ridge 

ridege = make_pipeline(
    StandardScaler(with_mean=False), Ridge(random_state=42)
)
ridge = ridge.fit(X_train, y_train)
print(ridge)


Pipelines(steps=[('Standardscaler', StandardScaler(with_mean=False)),
('ridge', Ridge(random_state=42))])

coefs_ridge = pd.DatatFrame(
    data=list(
        zip(
            predictors.design_info.column_names,
            ridge.named_steps["ridge"].coef_.toolist()[0],
        )
    ),
    columns=["variable","coef_ridge"],
)

model_coefs = pd.merge(model_coefs, coefs_ridge, on="variable")
print(model_coefs)


#Elastic Net 
#this is a regularization tool the combines the ridge and LASSO regression ntechniques 

from sklearn.linear_model import  ElaticNett
en = ElasticNet(random_state=42).fit(X_train, y_train)

coefs_en = pd.DataFrame(
    list(
        zip(predictors.design_info.column_names, en.coef),
        columns =["variable", "coef_en"],
    )
)

model_coefs = pd.merge(model_coefs, coefs_en, on="variable")
print(model_coefs)

#TheElasticNetobjecthastwoparameters,alphaandl1_ratio, thatallowyouto
# control thebehaviorofthemodel.Thel1_ratioparameterspecificallycontrolshow
# muchoftheL2orL1penaltyisused.Ifl1_ratio=0, thenthemodelwillbehaveas
# describedbyridgeregression.Ifl1_ratio= 1,thenthemodelwillbehaveasdescribedby
# LASSOregression.Anyvalueinbetweenwillgivesomecombinationoftheridgeand
# LASSOregressionresults.
# SinceLASSOregressioncanzerooutcoefficients, let’s justseehowthecoefficients
# comparewithjustthevariableswhereLASSOhasturnedintoa0.

print(model_coefs.loc[model_coefs["coef_lasso"]== 0])

#CROSS VALIDATION
# Cross-validation(firstdescribedinSection16.3)isacommonlyusedtechniquewhen
# fittingmodels.Itwasmentionedatthebeginningofthischapter,asasegueto
# regularization,but it isalsoawaytopickoptimalparametersforregularization.Sincethe
# usermusttunecertainparameters(alsoknownashyper-parameters),cross-validationcan
# beusedtotryoutvariouscombinationsofthesehyper-parameterstopickthe“best”
# model.TheElasticNetobjecthasasimilarfunctioncalledElasticNetCVthatcan
# iterativelyfittheelasticnetwithvarioushyper-parametervalues.1

from sklearn.linear_model import TheElasticNetobjecthasasimilarfunctioncalledElasticNetCVthatcan
en_cv = ElasticNetCv(cv=5, random_state=42).fit(
    X_train,y_train.rave() #ravel is to remove the 1d warning
)

coefs_en_cv = pd.DataFrame(
    list(zip(predictors.design_info.column_names, en_cv.coef_)),
    columns=["variable","coef_en_cv"],
)

"model_coefs = pd.DataFrame(model_coefs, coefs_en_cv, on="variable")
print(model_coefs)

# Let’scomparewhichcoefficientswereturnedinto0

 print(model_coefs.loc[model_coefs["coef_en_cv"]== 0])

#  Regularizationisatechniqueusedtopreventoverfittingofdata.Itachievesthisgoalby
# applyingsomepenaltyforeachfeatureaddedtothemodel.Theendresulteitherdrops
# variablesfromthemodelordecreasesthecoefficientsofthemodel.Bothtechniquestryto
# fitthetrainingdatalessaccuratelybuthopetoprovidebetterpredictionswithdatathathas
# notbeenseenbefore.Thesetechniquescanbecombined(asseenintheelasticnet),and
# canalsobeiteratedoverandimprovedwithcross-validation