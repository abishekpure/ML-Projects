# -*- coding: utf-8 -*-
"""ChicagoArrestRecords.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17DhkLlQydUyWqScUZ496-kj2Arff9bek
"""

import numpy as np 
import pandas as pd
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Chicago Arrest Records/chicagoarrestrecord.csv')
df.columns

df.shape

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_df = df.select_dtypes(include=numerics)
len(numeric_df.columns)

numeric_df.columns

df.head()

df['ARREST DATE'].dtype

df['ARREST DATE'] = pd.to_datetime(df['ARREST DATE'])

df['ARREST DATE'].dtype

df['ARREST DATE']

race_stat = df.groupby('RACE').count()
race_stat = race_stat.reset_index('RACE')
race_stat

race_stat_plot = sns.barplot(x='RACE', y='CB_NO', data=race_stat)
plt.setp(race_stat_plot.get_xticklabels(), rotation=90)
race_stat_plot.set(xlabel='Race', ylabel='Total Number of Arrest')
plt.show()

race_stat['PERCENTAGE'] = (race_stat['CB_NO']/len(df))*100
race_stat_penc = race_stat.groupby('RACE')['PERCENTAGE'].mean()
race_stat_penc

colors = sns.color_palette('bright')
#explode = (0.2, 0, 0.2,0,0.2,0,0.2)
plt.pie(race_stat_penc, colors=colors) #wedgeprops = {autopct='%.2f%%,'edgecolor':'white', 'linewidth':2, 'antialiased':True})
plt.legend(race_stat['RACE'], title='RACE', loc='center left', bbox_to_anchor=(2,0,0.5,1))
plt.axis('equal')
#plt.tight_layout()
plt.show()

df['YEAR'] = df['ARREST DATE'].dt.year
df['YEAR']

arrest_per_year = df.groupby('YEAR').count()
arrest_per_year = arrest_per_year.reset_index('YEAR')
arrest_per_year

arrest_per_year.groupby('YEAR')['CB_NO'].sum()

black_arrest_per_year_plot = sns.barplot(x='YEAR', y='CB_NO', data=arrest_per_year)
black_arrest_per_year_plot.set(ylabel='Total Number of Arrests Per Year')

sns.distplot(arrest_per_year)

black_arrest = df[df['RACE'] == 'BLACK']
len (black_arrest)

len(df)

150471-106957

black_arrest_per_year = black_arrest.groupby('YEAR')['CB_NO'].count()
black_arrest_per_year = black_arrest_per_year.reset_index('YEAR')
black_arrest_per_year

black_arrest_per_year_plot = sns.barplot(x='YEAR', y='CB_NO',data=black_arrest_per_year)
black_arrest_per_year_plot.set(ylabel='Total no of Arrests of Black Race people')

offence_category = df.groupby('CHARGE 1 TYPE').count()
offence_category = offence_category.reset_index('CHARGE 1 TYPE')
offence_category.groupby('CHARGE 1 TYPE')['CB_NO'].sum()

offence_category_graph= sns.barplot(x='CHARGE 1 TYPE', y='CB_NO', data=offence_category)
offence_category_graph.set(xlabel='Charge Type' ,ylabel='Number of Cases')