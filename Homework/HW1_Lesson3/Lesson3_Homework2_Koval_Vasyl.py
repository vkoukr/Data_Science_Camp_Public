import numpy as np
import pandas as pd
print(pd.__version__)

#-----------First question------------
def answer_one():
    file_name = 'Energy Indicators.xls'
    energy = pd.read_excel(file_name, sheet_name='Energy',
                           skiprows=18, header=None)
    energy = energy.iloc[:227]
    energy = energy.drop(energy.columns[[0, 1]], axis=1)
    energy.columns = ['Country', 'Energy Supply',
                      'Energy Supply per Capita', '% Renewable']
    energy['Energy Supply'] *= 1000000
    
    for i in range(energy.shape[0]):
        for j in range(1, energy.shape[1]):
            if not str(energy.iloc[i, 1]).isdigit():
                energy.iloc[i, j] = np.NaN
    
    
    def remove_digit(receive_country):
        for i in range(len(receive_country)):
          if receive_country[i].isdigit():
              # print('Before: ', receive_country)
              # print('After - ', receive_country[:i])
              return receive_country[:i]
          elif (receive_country[i] == '('):
              # print('Before: ', receive_country)
              # print('After - ', receive_country[:i-1])
              return receive_country[:i-1]
        return receive_country
    
    
    energy['Country'] = energy['Country'].apply(remove_digit)
    
    energy['Country'].replace({
        'Republic of Korea': 'South Korea',
        'United States of America': 'United States',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'China, Hong Kong Special Administrative Region': 'Hong Kong'}, inplace=True)
    
    #----------- Reading from CSV ------------
    GDP = pd.read_csv('world_bank.csv', skiprows=4)
    GDP['Country Name'].replace({
        'Korea, Rep.': 'South Korea',
        'Iran, Islamic Rep.': 'Iran',
        'Hong Kong SAR, China': 'Hong Kong'}, inplace=True)
    GDP.rename(columns={'Country Name': 'Country'}, inplace=True)
    
    #----------- Reading from XLS ------------
    file_name = 'scimagojr country rank 1996-2020.xlsx'
    ScimEn = pd.read_excel(file_name)
    merged1 = pd.merge(energy, GDP, on='Country')
    merged2 = pd.merge(merged1, ScimEn, on='Country')
    merged2 = merged2[(merged2['Rank'] > 0) & (merged2['Rank'] < 16)]
    merged2.set_index('Country', inplace=True)
    merged2 = merged2.loc[:, ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index',
                              'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
    merged2 = merged2.sort_values('Rank', ascending=False)
    return merged2

print('\n======ANSWER 1======\n',answer_one())


#--------- Second Question-----------------
def answer_two():
    Top15 = answer_one()
    avgGDP =Top15.iloc[:,10:].mean(axis = 1)
    avgGDP.name=('avgGDP')
    avgGDP.sort_values(ascending= False, inplace=True)
    return pd.Series(avgGDP)

print('\n======ANSWER 2======\n',answer_two())

#--------- Third Question-----------------
def answer_three():
    # Top15 = answer_one()
    # Top15['AvgGDP'] = answer_two()
    # Top15.sort_values('AvgGDP', ascending=False, inplace=True)
    # ANSWER=Top15.iloc[5,'2015']-Top15.iloc[5,'2006']
    # print(ANSWER)
    Top15 = answer_one()
    AverageDF = answer_two()
    t=AverageDF.index[5]
    ANSWER=Top15.loc[t,'2015']-Top15.loc[t,'2006']
#    print(abs(ANSWER))
    return ANSWER

a=answer_three()
print('\n======ANSWER 3======\n',answer_three())

#--------- Fourth Question-----------------
def answer_four():
    Top15 = answer_one()
    Top15['Ratio_Citation'] = Top15['Self-citations'] / Top15['Citations']
    Country_Citation = Top15.index[Top15['Ratio_Citation'].argmax()]
    Max_Ratio = Top15['Ratio_Citation'].max()
    #print(Top15.index[Top15['Ratio_Citation'].argmax()],
    Top15['Ratio_Citation'].max()
    ANSWER=Country_Citation, Max_Ratio
    return ANSWER

a=answer_four()
print('\n======ANSWER 4======\n',answer_four())

#--------- Fiveth Question-----------------
def answer_five():
    Top15 = answer_one()
    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15.sort_values('Population', ascending=False, inplace=True)
    ANSWER=Top15.index[2]
    return ANSWER
a=answer_five()
print('\n======ANSWER 5======\n',answer_five())

#--------- Sixth Question-----------------
def answer_six():
    Top15 = answer_one()
    Top15['People_numb'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['CitablePerCapita'] = Top15['Citable documents'] / Top15['People_numb'].astype(float)
    ANSWER=Top15['CitablePerCapita'].corr(Top15['Energy Supply per Capita'].astype(float), method='pearson')
    return ANSWER
a=answer_six()
print('\n======ANSWER 6======\n',answer_six())

#--------- Seven Question-----------------
def answer_seven():
    Dict_to_analize = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = answer_one()
    Top15['People_numb'] = Top15['Energy Supply']/Top15['Energy Supply per Capita']   
    
    # for new_DataFrame, frame in Top15.groupby(Dict_to_analize):
    #     new_DataFrame.loc[new_DataFrame] = [len(frame), frame['People_numb'].sum(),frame['People_numb'].mean(),frame['People_numb'].std()]
    # return new_DataFrame

    Top15['Continent'] = None
    for i in range(len(Top15)):
        Top15.iloc[i,21]= Dict_to_analize[Top15.index[i]]
    new_DataFrame = Top15['People_numb'].astype(float).groupby(Top15['Continent']).agg(['size', 'sum', 'mean', 'std'])
    return new_DataFrame

a=answer_seven()
print('\n======ANSWER 7======\n',answer_seven())