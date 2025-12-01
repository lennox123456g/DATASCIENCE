#for binary data like sick ad not sick
#count data like how many coins will get if i flip the coin 
#GENERALISED LINEAR MODELS CAN E USED IN THIS CASE 
#but still a linear combination of  predictors is used 


#SO LETS START WITH 
#LOGISTIC REGRESSION (BINARY OUTCOME)
# When you have a binary response variable (i.e., two possible outcomes), logistic
# regression is often used to model the data. 

#Example using the titanic dataset 

import seaborn  as sns 

titanic = sns.load_data set("titanic")
titanic.to_csv("data/titanic.csv", index=False)


#Now lets subset our data to have columns we are to use here 
#.Wewillalsobedroppingrowswithmissingvaluesinthemsince
# modelsusuallyignoreobservationsthatarenotcompleteanyway,andwearenotshowing
# howtoimputemissing data in this chapter.Notice that we are dropping the missing values
# after wesubsettedthecolumnswewanted,sowearenotartificially dropping observations.

titanic_sub = (
    titanic[["survived", "sex","embarked"]].copy().dropna()
)

print(titanic_sub)


# In this dataset,our outcome of interest is the survived column,on whether an
# individual survived (1) ordied (0)during the sinkingo f the Titanic.The other columns,
# sex,age,ande mbarked are going to be the variable we use to see who survived.


#prnt out the values in the survived column 
print(titanic_sub["survived"].value_counts())

#embarked column is for whee the individuals boarded the ship from 
#Southampton(S),Cherbourg(C),andQueenstown(Q)


#count of values in the embarked column
print(titanic_sub["embarked"].value_counts())

# Interpreting results from a logistic regression model is not as straightforward as
# interpreting a linear regression model. In a logistic regression, as with all generalized linear
# models, there is a transformation (i.e., link function), that that affects how to interpret
# the results.

#Where p is the probability of the event, and p
 )
 ( p
 1 −p
# 1−p is the odds of the event. This is why
# logistic regression output is typically interpreted as “odds,” and we do that by undoing the
# log call by exponentiating our results. You can think of the “odds” of something as how
# many “times likely” the outcome will be. That phrasing should only be used as an
# analogy, however, as it is not technically correct. The value of an odds can only be greater
# than zero, and can never be negative. However, the “log odds” (i.e., logit), can be negative.


WITH STATS MODELS 
#WE USE THE LOGIT()  FUNCTION 
#sysntx i slie for tht used in linear regression 


import statsmodels.formula.api as smf 

#formula for the model 
form = 'survived ~ sex + age + embarked '

#fitting the logistic regression model, note the fit() at the ned 
py_logistic_smf = smf.logit(formula=form, date=titanic_sub).fit()

print(py_logistic_smf.summary())

#We can then get the coefficients of the model,and exponentiate it to calculate the odds
#of each variable

import numpy as np
#get the coeffecients into  a dataframe
re_sm = pd.DataFrame(py_logistic_smf.params, columns =["coefs_sm"])


#round the decimals 
print(re_sm.round(3))

# Anexampleinterpretationofthesenumberswouldbethatforeveryoneunit increase
# inage,theoddsofthesurviveddecreasesby0.992times.Sincethevalueiscloseto1, it
# seemsthatagewasn’ttoomuchofafactorinsurvival.Youcanalsoconfirmthatstatement
# bylookingatthep-valueforthevariableinthesummarytable(undertheP>|z|column).
# Asimilarinterpretationcanbemadewithcategoricalvariables.Recall thatcategorical
# variablesarealwaysinterpretedinrelationtothereferencevariable.
# Therearetwopotentialvaluesforsexinthisdataset,maleandfemale,butonlya
# coefficientformaleisgiven.Sothatmeansthevalueisinterpretedas“malescomparedto
# females,”wherefemaleisthereferencevariable.Theoddsforthemalevariableare
# interpretedas:maleswere0.084timesmorelikelytosurvivecomparedtofemales(the
# odds fornotsurvivingthetragedywerehighformales)


#WITH SKLEARN 
#REMEBER THAT WITH SKLEARN ,DUMMY VARIABLES NEED TO BE CREATED MANUALLY 

titanic_dummy = pd.get_dummies(
    titanic_sub[["survived ", "sex", "age", "embarked"]],
    drop_first = True
)

 #note our outcome variable is the firstcolumn(index0)
 print(titanic_dummy)

 #now we can use the LOgisticrEGRESSION() function from 
 #the linear_model module to create a logistic regression output to fit our model

 from sklearn import linear_model

 #this is the only part that fits the model 

 py_logistic_sklearn1= (
    linear_model.LogisticRegression().fit(
        X=titanic_dummy.iloc[:, 1:], #all columns except the first
        y=titanic_dummy.iloc[:,0] #justthe first column
    )
 )

 #the code below will help us to to process the scikit learn
 #logistic regression fitted model into a  single dataframe so we can better compare results 

 #get the names 0f the dummy  variable columns 
 dummy_names = titanic_dummy.columns.to_list()


 #get the intercept and coeffecients into a dataframe 
 sk1_res1 = pd.DataFrame(
    py_logistic_sklearn1.intercept_,
    index=["Intercept"],
    columns=["coeff_sk1"],
 )
sk1_res2 = pd.DtaFrame(
    py_logistic_sklearn1.coeff_.T,
    index= dummy_names[1:],
    columns=["coef_sk1"],
)

#piut the results into a single DataFrame to show the results 
res_sklearn_pd_1= pd.concat([sk1_res1,sk1_res2])

#calculate the odds 
res_sklearn_pd_1["odds_sk1"] = np.exp(res_sklearn_pd_1["coef_sk1"])

print(res_sklearn_pd_1.round(3))

# You will notice here that the coefficient values are different from the ones calculated
# from the statsmodels section we just did. The differences are more than a simple
# rounding error too!


#BEING CAREFUL OF SCIKIT LEARN 
# The main reason why the sklearn results differ from the statsmodels results stems from
# the domain differences where the two packages come from. Scikit-learn comes more from
# the machine learning world and is focused on prediction so the model defaults are set for
# numeric stability, and not for inference. However, statsmodels functions are implemented
# in a manner more traditional for statistics.
# The LogisticRegression() function has a penalty parameter that defaults to 'l2',
# which adds an L2 penalty term (more about penalty terms in Chapter 17). If we want
# LogisticRegression() to behave in a manner more traditional for statistics, we need to set
# penalty="none".

#Fit another Logistic regression with no penalty 
py_logistic_sklearn2= linear_model.LogisticRegression(
    penalty=none  #this parameter is important 
).fit(
    X=titanic_dummy.iloc[:, 1:], # all columns except first 
    y=titanic_dummy.iloc[:,0]  #only the first column 
)

 # rest of the code is the same as before, except variable names
 sk2_res1 = pd.DataFrame(
    py_logistic_sklearn2.intercept_,
    index=["Intercept"],
    columns=["coef_sk2"],
 )
 sk2_res2 = pd.DataFrame(
    py_logistic_sklearn2.coef_.T,
    index=dummy_names[1:],
    columns=["coef_sk2"],
    )
 res_sklearn_pd_2 = pd.concat([sk2_res1, sk2_res2])
 res_sklearn_pd_2["odds_sk2"] = np.exp(res_sklearn_pd_2["coef_sk2"])


 #MAKE SURE YOU KNOW THE FUNCTIONS DOCUMNATATION AND WHAT THEY ARE DOING 


 #LOOKING AT TE ORIGINAL STATSMODODEL RESULTS 
 sm_results = re_sm.round(3)

 #sort the values to make things easier to compare 
 sm_results = sm_results.sort_index()

 print(sm_results)


 #NOW LETS COMPARE THEM WITH THE TWO SKLEARN RESULTS 
 # concatenate the 2 model results
 sk_results = pd.concat(
 [res_sklearn_pd_1.round(3), res_sklearn_pd_2.round(3)],
 axis="columns",
 )
 # sort cols and rows to make things easy to compare
 sk_results = sk_results[sk_results.columns.sort_values()]
 sk_results = sk_results.sort_index()
 print(sk_results)



#POISSON REGRESSION (COUNT OUTCOME VARIABLE)
#THIS IS USED WHEN oUR response  VARIABLE INVOLVES COUNT DATA 
acs = pd.read_csv('data/acs_ny.csv')
print(acs.columns)

#About the ACS Data Set
# The American Community Survey (ACS) data we are using contains information about
# family and house size in New York


#with STATSMODELS 
#we can perform the poisson function using poisson() function in stats models 
# We will use the NumBedrooms variable

import matplotlib.pyplot as plt 

fig,ax = plt.subplots()
sns.countplot(data=acs, x="NumBedrooms", ax=ax)

ax.set_title("Number of Bedrooms")
ax.set_xlabel('Number of Bedrooms in a House ')
ax.set_ylabel('count')

plt.show()

model = smf.poisson(
    "NumBedrooms ~ HouseCosts +OwnRent", data=acs
)
results = model.fit()

print(results.summary())

#MERITS OF USING GENERALISED LINEAR MODELS 
# The benefit of using a generalized linear model is that the only things that need to be
# changed are the family of the model that needs to be fit, and the link function that
# transforms our data. We can also use the more general glm() function to perform all the
# same calculations.

import statsmodels.api as sm 
import statsmodels.formula.api as smf
model = smf.glm(
    "NumBedrooms ~ HouseCosts + OwnRent",
    data=acs,
    family=sm.families.Poisson(sm.genmod.families.limks.log())
).fit()

# In this example, we are using the Poisson family, which comes from sm.families.
# Poisson, and we’re passing in the log link function via sm.genmod.families.links.log().
# We get the same values as we did earlier when we use this method.

print(results.summary())

#NEGATIVE BINOMIAL REGRESSION FOR DISPERSION 
# If our assumptions for Poisson regression are violated—that is, if our data has
# overdispersion—we can perform a negative binomial regression instead (Figure 14.2).
# Overdispersion is the statistics term meaning the numbers have more variance than
# expected, i.e., the values are too spread out.

fig, ax = plt.subplots()
sns.countplot(data= acs, x="NumPeople, ax=ax")
 ax.set_title N('Number of People')
 ax.set_xlabel('Number of People in a Household ')
 ax.set_ylabel('Count')
 plt.show()

 model = smf.glm(
    "NumPeople ~ Acres + NumVehicles",
    data=acs,
    family=sm.families.NegativeBinomial(
        sm.genmod.families.links.log()
    ),
 )

 results = model.fit()
 print(results.summary())

#  Look for the reference variable in Acres.
 print(acs["Acres"].value_counts())

#  MORE GENERALISED MODELS CHECK THE GLM documentation in statsmodels 
#  Binomial . Gamma . Gaussian . InverseGaussian . NegativeBinomial . Poisson . Tweedie
# Thelinkfunctionsarefoundundersm.families.family.<FAMILY>.links.Followingis
# thelistof linkfunctions,butnotethatnotall linkfunctionsareavailableforeachfamily:
# . CDFLink . CLogLog . LogLog . Log . Logit . NegativeBinomial . Power . cauchy . cloglog . loglo
#identity . inverse_power . inverse_squared . log . logit
   For example, using the all the link functions for the Binomial family.
 sm.families.family.Binomial.links
 [statsmodels.genmod.families.links.Logit,
 statsmodels.genmod.families.links.probit,
 statsmodels.genmod.families.links.cauchy,
 statsmodels.genmod.families.links.Log,
 statsmodels.genmod.families.links.CLogLog,
 statsmodels.genmod.families.links.LogLog,
 statsmodels.genmod.families.links.identity]


 #SURVIVAL ANALYSIS 
  #Survival analysis is used when we want to model how much time passes before something
# happens. It is typically used in health contexts when we are looking to see if a drug or
# intervention prevents an adverse event from occurring. Before we begin with examples of
# survival analysis, let’s define some terms first.
# . Event: Outcome, situation, or “event” you are interested in tracking in your study.
# . Follow-up: “Lost to follow-up” is a term used in medical data. It means that the
# patient stopped “following up” to the visits. This can mean that the patient just
# stopped showing up, or the patient has died Usually, in this context, death is the
# “event” of interest.
# . Censoring: Unsure of the status for a particular observation. This can be
# right-censored (no more data after this period of time), or left-censored (no data
# before this period of time). Right-censoring typically occurs from lost to follow up,
# or the event of interest has occurred (e.g., death).
# . Stop time: A point in the data where some censoring event has occurred.
# Survival analysis is typically used in medical research when trying to determine whether
# one treatment prevents a serious adverse event (e.g., death) better than the standard or a
# different treatment. Survival analysis is also used when data is censored, meaning the exact
# outcome of an event is not entirely known. For example, patients who follow a treatment
# regimen may sometimes be lost break to follow-up. The censoring usually occurs at a
# “stop” event.
# Survival analysis is performed using the lifelines library.1

#SURVIVAL DATA    #PAGE 312
bladder = pd.read_csv('data/bladder.csv')
print(bladder)

#ABOUT THE BLADDER DATSET 
#About the Bladder Data Set
# The bladder data set comes from the R {survival} package. It contains 85 patients, their
# cancer recurrence status, and what treatment they were on. Below is a recreation of the
# code book for the data.

print(bladde['rx'].value_counts())

#KEPLAN MEIR CURVES FOR PERORMING OUR SURVIVAL ANALYSIS 
# To perform our survival analysis, we import the KaplanMeierFitter() function from the
# lifelines library

from lifelines import KaplanMeierFitter

#Creating the model and fitting the data proceeds similarly to how models are fit using
# sklearn. The stop variable indicates when an event occurs, and the event variable signals
# whether the event of interest (bladder cancer re-occurrence) occurred. The event value
# can have a value of , because people can be lost to follow-up. As noted earlier, this type of
# data is called “censored.”

kmf = KaplanMeierFitter()
kmf.fit(bladder['stop'], event_observed=bladder['event'])

# We can plot the survival curve using matplotlib, 
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
kmf.survival_function_.plot(ax=ax)
ax.set_title('Surval function of cancer recurrence')
plt.show()

#We can also show the confidence interval of our survival curve,
fig, ax = plt.subplots()
kmf.plot(ax = ax)
ax.set_title('Survival with confidence intervals ')
plt.show()


#COX PROPORTIONAL HAZARDS 
# So far, we’ve just plotted the survival curve. We can also fit a model to predict survival rate.
# One such model is called the Cox proportional hazards model. We fit this model using the
# CoxPHFitter
# class from lifeline

from lifelines import CoxPHFitter
cph = CoxPHFitter()

#we then pass in the columns to be used as predictors 
cph_bladder_df = bladder [
    ["rx", "number","size", "enum","stop","event"]
]
cph.fit(cph_bladder_df ,duration_col="stop", event_col="event")

# Now we can use the .print_summary method to print out the coefficients.
cph.print_summary()

#We mainly focus on the hazard ratio when looking at CPH models. In the table this is#
# represented by the exp(coef) column in the results. Values close to 1 show that there is no
# change in the survival hazard. Values from 0-- 1 show a smaller hazard and values greater
# than 1 show an increase in hazard.

#TESTING THE COX MODEL ASSUMPTIONS 
#One way to check the Cox model’s assumptions is to plot a separate survival curve by
# strata. In our example, our strata will be the values of the rx column, meaning we will plot
# a separate curve for each type of treatment. If the log(-log(survival curve)) versus
# log(time) curves cross each other (Figure 15.3), it signals that the model needs to be
# stratified by the variable.
 rx1 = bladder.loc[bladder['rx'] == 1]
 rx2 = bladder.loc[bladder['rx'] == 2]
 kmf1 = KaplanMeierFitter()
 kmf1.fit(rx1['stop'], event_observed=rx1['event'])
 kmf2 = KaplanMeierFitter()
 kmf2.fit(rx2['stop'], event_observed=rx2['event'])
 fig, axes = plt.subplots()
 # put both plots on the same axes
 kmf1.plot_loglogs(ax=axes)
 kmf2.plot_loglogs(ax=axes)

 axes.legend('rx1','rx2')
 plt.show()

# Since the lines cross each other, it makes sense to stratify our analysis.
cph_strat = COxPHFitter()
cph_strat.fit(
    cph_bladder_df,
    duration_col="stop",
    event_col="event",
    strata=["rx"]
)
cph_strat.print_summary()

# Survival models measure “time to event” with censoring. They are commonly used in a
# health context but do not have to be solely used in that domain. If you can define some
# kind of event of interest, e.g., people who come to my website and purchase an item, you
# can potentially use survival models