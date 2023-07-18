import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
%matplotlib inline
import os

for dirname, _, filenames in os.walk('/Users/prabalkalhans/Desktop/Crop Yeild Production'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

file_loc = "/Users/prabalkalhans/Desktop/Crop Yeild Production/Project4_Ag_Prediction of Agriculture Crop Production In India/"
file_name = "datafile.csv"
crops_data = pd.read_csv(file_loc+file_name)
crops_data.info()

print(crops_data)

rops_data = crops_data.dropna()

print(crops_data)

plt.figure(figsize=(10,8))
sns.scatterplot(data = crops_data)
plt.xlabel('Crops')
plt.ylabel('Level of Production')

plt.figure(figsize=(15,10))
sns.boxplot(data=crops_data)
plt.xlabel('Year-wise Production')
plt.ylabel('Level of Production')

file_loc = '/Users/prabalkalhans/Desktop/Crop Yeild Production/Project4_Ag_Prediction of Agriculture Crop Production In India/'
file_name = 'datafile.csv'
crops_data = pd.read_csv(file_loc + file_name)

cdata = pd.read_csv(file_loc+file_name)
cdata = cdata.dropna()

plt.figure(figsize=(15,15))
columns = cdata.columns
y = cdata.columns[1:]

for col in y:
    plt.plot(cdata['Crop'], cdata[col])
    plt.xlabel('Crop')
    plt.legend(y)


file_name = 'datafile (1).csv'
cult_data = pd.read_csv(file_loc+file_name)
print(cult_data.info())

print(cult_data.head())

print(cult_data['Crop'].unique())
print(cult_data['State'].unique())

temp = pd.crosstab(cult_data['State'],cult_data['Crop'])

temp.plot(kind='bar', stacked=True, figsize = (20,10))

file_name = 'datafile (1).csv'
cult_data = pd.read_csv(file_loc+file_name)
print(cult_data.info())

print(cult_data.head())

print(cult_data.columns)

file_name = 'datafile (2).csv'
grains_data = pd.read_csv(file_loc+file_name, sep=',')
print(grains_data.info())

grains_data.columns

grains_data['Crop'] = grains_data['Crop             ']

del grains_data['Crop             ']

prod_cols = ['Crop','Production 2006-07', 'Production 2007-08', 'Production 2008-09','Production 2009-10', 'Production 2010-11']
area_cols = ['Crop','Area 2006-07', 'Area 2007-08', 'Area 2008-09','Area 2009-10', 'Area 2010-11']
yield_cols = ['Crop','Yield 2006-07', 'Yield 2007-08', 'Yield 2008-09','Yield 2009-10', 'Yield 2010-11']
prod_data = grains_data[prod_cols]
area_data = grains_data[area_cols]
yield_data = grains_data[yield_cols]

prod_data.index = prod_data['Crop']
del prod_data['Crop']

area_data.index = area_data['Crop']
del area_data['Crop']

yield_data.index = yield_data['Crop']
del yield_data['Crop']


data = pd.read_csv("/Users/prabalkalhans/Desktop/Crop Yeild Production/Project4_Ag_Prediction of Agriculture Crop Production In India/datafile (1).csv")

fig,axs = plt.subplots(figsize=(10,6))
crop_wise_yield = data.groupby(['Crop']).sum()['Yield (Quintal/ Hectare) ']
plt.plot(crop_wise_yield)
crop_wise_production = data.groupby(['Crop']).sum()['Cost of Production (`/Quintal) C2']/10
plt.plot(crop_wise_production)
plt.xticks(rotation ='vertical')
plt.legend()

import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/Users/prabalkalhans/Desktop/Crop Yeild Production/Project4_Ag_Prediction of Agriculture Crop Production In India'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv("/Users/prabalkalhans/Desktop/Crop Yeild Production/Project4_Ag_Prediction of Agriculture Crop Production In India/datafile (2).csv")

import seaborn as sns
import matplotlib.pyplot as plt
df

df.columns = ['Crop', 'Production 2006-07', 'Production 2007-08',
       'Production 2008-09', 'Production 2009-10', 'Production 2010-11',
       'Area 2006-07', 'Area 2007-08', 'Area 2008-09', 'Area 2009-10',
       'Area 2010-11', 'Yield 2006-07', 'Yield 2007-08', 'Yield 2008-09',
       'Yield 2009-10', 'Yield 2010-11']

plt.figure(figsize=(20,8))
sns.lineplot(data=df,x="Crop",y="Production 2006-07",marker='o')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,8))
sns.lineplot(data=df,x="Crop",y="Area 2006-07",marker='o')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,8))
sns.lineplot(data=df,x="Crop",y="Yield 2006-07",marker='o')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,8))
sns.barplot(data=df,x="Crop",y="Production 2006-07",color='lightgreen')
sns.lineplot(data=df,x="Crop",y="Area 2006-07",marker='o',label='Area 2006-07')
sns.lineplot(data=df,x="Crop",y="Yield 2006-07",marker='o',label='Yield 2006-07')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,8))
sns.lineplot(data=df,x="Crop",y="Production 2010-11",marker='o', color= 'red')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,8))
sns.lineplot(data=df,x="Crop",y="Area 2010-11",marker='o', color= 'red')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,8))
sns.lineplot(data=df,x="Crop",y="Yield 2010-11",marker='o', color= 'red')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,7))
sns.barplot(data=df,x="Crop",y="Production 2010-11",color='lightgrey')
sns.lineplot(data=df,x="Crop",y="Area 2010-11",marker='o',label='Area 2010-11')
sns.lineplot(data=df,x="Crop",y="Yield 2010-11",marker='o',label='Yield 2010-11')
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(20,7))
sns.lineplot(data=df,x="Crop",y="Production 2006-07",marker='o',label='Production 2006-07', color= 'blue')
sns.lineplot(data=df,x="Crop",y="Production 2010-11",marker='o',label='Production 2010-11', color= 'black')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,7))
sns.lineplot(data=df,x="Crop",y="Area 2006-07",marker='o',label='Area 2006-07', color= 'red')
sns.lineplot(data=df,x="Crop",y="Area 2010-11",marker='o',label='Area 2010-11', color= 'black')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(20,7))
sns.lineplot(data=df,x="Crop",y="Yield 2006-07",marker='o',label='Yield 2006-07', color= 'black')
sns.lineplot(data=df,x="Crop",y="Yield 2010-11",marker='o',label='Yield 2010-11', color= 'gold')
plt.xticks(rotation=90)
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = crop_data.drop('Frequency', axis=1)  
y = crop_data['Particulars'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
