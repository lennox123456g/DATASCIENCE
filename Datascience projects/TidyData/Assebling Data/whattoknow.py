##combining various datasets for analysis 
#asses whether data was joined properly
#asses a single dataset from multiple files #use an appropriate functionn or method to combine multiple datasets 
#identify whether needs to be concatenated or joined together 

#   CONCATENATION 
#LIKE ADDING A ROW OR COLUMN TO DATA 
#LIKE SOMETHING YOU CALculated already 

#we use the comcat() function of pandas 
#Example 
import pandas as pd
df1 = pd.read_csv('data/concat_1.csv')
df2 = pd.read_csv('data/concat_2.csv')
df3 = pd.read_csv('data/concat_3.csv')

print(df1)
print(df2)
print(df3)

#test out
 print(df1.values)
 print(df1.index)
 print(df1.columns)

 #concatentaing seriously 
 row_concat = pd.concat([df1, df2,df3])
 print(row_concat)

 #trying to subset 
# subset the fourth row of the concatenated dataframe
print(row_concat.iloc[3, :])


#concatenating a row 
#using series will fail 
example 

 # create a new row of data
 new_row_series = pd.Series(['n1', 'n2', 'n3', 'n4'])
 print(new_row_series)

# attempt to add the new row to a dataframe
print(pd.concat([df1, new_row_series]))


#create a new column and mulitole nan values 

#so we use a dtatframe to add the row as shown below 

new_row_df = pd.DataFrame(
    #note the double brackets to create a "row" of data
    data = [['n1','n2', 'n3','n4']],
    columns = ["A". "B", " C", "D"],
) #note the colmns named here the values will go 
print(new_row_df)

#concatenate the rowofdata
print(pd.concat([df1,new_row_df]))


#ignore_INDEX CONCATENATION
#using ignore_index to reste row  index after concatenation 
 row_concat_i = pd.concat([df1,df2,df3],ignore_index=True)
 print(row_concat_i)


 #AVOIBING REPAEATING COLUM LABELAS 
 print(pd.concat([df1,df2,df3], axis="columns", ignore_index=True))


 #IF COLUMNS HAVE UNCORDINATED NAMES 
  #renamethe columnsofourdataframes
 df1.columns=['A','B','C','D']
 df2.columns=['E','F','G','H']
 df3.columns=['A','C','F','H']

 print(df1)
 print(df2)
 print(df3)


#we get nan allover 
 row_concat = pd.concat([df1, df2, df3])
 print(row_concat)


#AVOID NAN VALUES USING JOIN inner 
# A parameter named join
# accomplishes this. By default, it has a value of 'outer', meaning it will keep all the
# columns. However, we can set join='inner' to keep only the columns that are shared
# among the data sets.

# If we try to keep only the columns from all three dataframes, we will get an empty
# dataframe, since there are no columns in common.
 print(pd.concat([df1, df2, df3], join='inner'))

#what works  
 # Ifweusethedataframesthathavecolumnsincommon,onlythecolumnsthatallof
# themsharewillbereturned.
 print(pd.concat([df1,df3],ignore_index=False,join='inner'))


 #CONCATENATING COLUMNS WITH DIFFERENT ROWS


df1.index=[0,1,2,3]
df2.index=[4,5,6,7]
df3.index=[0,2,5,7]

 print(df1)
 print(df2)
 print(df3)

 #concatenatine via columns causes Nan values 
 col_concat = pd.concat([df1, df2, df3], axis="columns")
 print(col_concat)

 #so we sp[eacify  that only when there are matchig indices 
print(pd.concat([df1, df3], axis="columns", join='inner'))


#MULTIPLE DATA TAHT HAS BEEN SPLIT TO BE WORLED ON TOGETER 
 from pathlib import Path
 # from my current directory fine (glob) the this pattern
 billboard_data_files=(
 Path(".")
 .glob("data/billboard-by_week/billboard-*.csv")
 )
 #this line is optional if you want to see the full list of files
 billboard_data_files=sorted(list(billboard_data_files))
 print(billboard_data_files)

  #The type() ofbillboard_data_files is a generator object,so if you“ useit” youw ill
 lose its contents.If you want to see the full list,you would need to run:
 billboard_data_files=list(billboard_data_files)

 #TRYING TO LOAD THE FILES INDIVIDUALLY 
 billboard01= pd.read_csv(billboard_data_files[0])
 billboard02= pd.read_csv(billboard_data_files[1])
 billboard03= pd.read_csv(billboard_data_files[2])

 #justlook atoneofthe datasetsweloaded
  print(billboard01)

print(billboard01.shape)
print(billboard02.shape)
print(billboard03.shape)

 # concatenate the dataframes together
 billboard = pd.concat([billboard01, billboard02, billboard03])
 # shape of final concatenated taxi data
 print(billboard.shape)

 #CHECKING THE COCATENATIO USING ASSERT
assert (
    billboard01.shape[0]
    + billboard02.shape[0]
    + billboard03.shape[0]
    == billboard.shape[0]
 )


 USING LOOPS TO LOAD MULYIPLE FILES 
 # this part was the same as earlier
 from pathlib import Path
    billboard_data_files = (
    Path(".")
    .glob("data/billboard-by_week/billboard-*.csv")
 )
 # create an empty list to append to
 list_billboard_df = []
 # loop though each CSV filename
 for csv_filename in billboard_data_files:
 # you can choose to print the filename for debugging
 # print(csv_filename)
 # load the CSV file into a dataframe
 df = pd.read_csv(csv_filename)
 # append the dataframe to the list that will hold the dataframes
 list_billboard_df.append(df)
 # print the length of the dataframe
 print(len(list_billboard_df))


  #Important
# The Path.glob() method returns a generator (Appendix P). This means that when we
# go through each element of the “list,” the item gets “used up,” so it won’t exist again.
# This saves a lot of compute resources since Python does not need to store everything in
# memory all at once. The downside is you will need to re-create the generator if you
# plan on using it multiple times. You can opt to turn the generator into a regular python
# list so all the elements are stored perpetually by using the list() function, e.g.,
# list(billboard_data_files).

#LOADING MULTIPLE FILES USING A LIST COMPREHENSION
#we have to recreate the generator becausee  we 
#we used it up in the previoous example 
billboard_data_files = (
    Path(".")
    .glob("data/billboard-by_week/billboard-*.csv")
)

#the loop code without comments 
list_billboard_df = []
for csv_filename in billboard_data_files:
    df = pd.read_csv(csv_filename
    list_billboard_df.append(df)

    billboard_data_files=(
        Path(.)
        glob("data/billboard-by_week/billboard-*.csv")
    )
)

 # same code in a list comprehension
 billboard_dfs = [pd.read_csv(data) for data in billboard_data_files]

#Finally, we can concatenate the results just as we did earlier.
 billboard_concat_comp = pd.concat(billboard_dfs)


# Warning
# If you get a ValueError: No objects to concatenate message, it means you did not
# re-create the billboard_data_files generator.
#print(billboard_concat_comp)


#mERGING IS WHTS THE JOIN USES ,
#WE CAN AS WELL be explicit with it 
person = pd.read_csv('data/survey_person.csv')
 site = pd.read_csv('data/survey_site.csv')
 survey = pd.read_csv('data/survey_survey.csv')
 visited = pd.read_csv('data/survey_visited.csv')


print(site)
print(visited)
print(survey)

#Exaamples 
 o2o_merge = site.merge(
 visited_subset, left_on="name", right_on="site"
 )
 print(o2o_merge)


#learn more about One to One merge 
#many to one merge 
#many to many merge 

#Danger
# All thecodeforperformingamergeusesthesamemethod,.merge().Theonlything
# thatmakestheresultsdifferiswhetherornottheleftand/orrightdataframehas
# duplicatekeys.
# Inpractice,youusuallydonotwantamany-to-manymerge.Sincethatmeansa
# cartesianproductofthekeyswerejoinedtogether.That is,everycombinationof
#duplicatedvalueswerecombined.


#ASSERTING 
# A simple way to check your work before and after a merge is by looking at the number of
# rows of our data before and after the merge. If you end up with more rows than either of
# the dataframes you are merging together, that means a many-to-many merge occurred,
# and that is usually situation you do not want.

 # expect this to be true
 # note there is no output
 assert vs.shape[0] == 21

 #Ech type of Observational unit forms a table


 #NORMALISATION 
# Its like the opposite of preparing data for analysis , visulization and mode fitting 
#we mely what we had made long
#we separate the repeated ones on which otehrs are categorised 