import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("world_population_data.csv")
st.title(" Welcome to Dashboard of world population EDA :bar_chart: :world_map: ")
st.write("by: H.Marques")

with st.container():
    st.header("Importing the packages:")
    code = '''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

            '''
    st.code(code, language='python')

with st.container():
    st.header("Reading the CSV:")
    code = '''
    df = pd.read_csv("world_population_data.csv")
            '''
    st.code(code, language='python')
    df
with st.container():
    st.header("Reading the 5 first values:")
    code = '''
    df.head()
            '''
    st.code(code, language='python')
    df[0:5]
with st.container():
    st.header("Seeing the dataframe shape:")
    code = '''
    df.shape
            '''
    st.code(code, language='python')
    df.shape
with st.container():
    st.header("Seeing null count and dtype of the columns:")
    code = '''
    df.info()
            '''
    st.code(code, language='python')
    df.info()
with st.container():
    st.header("Making sure that the columns don't have null values:")
    code = '''
    for col in range(len(df.columns)):
        if df[df.columns[col]].isnull().all() == False :
            print(f"There are NOT null values in {df.columns[col]}")
        else:
            print(f"There ARE null values in {df.columns[col]}")
            '''
    st.code(code, language='python')
    for col in range(len(df.columns)):
        if df[df.columns[col]].isnull().all() == False :
            st.write(f"There are NOT null values in {df.columns[col]}")
        else:
            st.write(f"There ARE null values in {df.columns[col]}")
    st.markdown("As we can see, the dataset don't have any nullable value, so let's continue our analysis.")
with st.container():
    st.header("Now we will see some values of the dataframe:")
    code = '''
    df
            '''
    st.code(code, language='python')
    df
    st.markdown("There are so many higher values from population for China and India, for exemple, in comparisson of the anothers countries. So is expected that there are some outliers.")
with st.container():
    st.header("Seeing that outliers in a boxplot:")
    code = '''
    fig, ax = plt.subplots()
    sns.boxplot(data=df)
    xticks_ = plt.xticks()
    print(xticks_)
    ax.set_xticks(xticks_[0],xticks_[1],rotation=90)
    ax.set_title("Boxplots for see the proportion of outliers")
    plt.show()

            '''
    st.code(code, language='python')
    fig, ax = plt.subplots()
    sns.boxplot(data=df)
    xticks_ = plt.xticks()
    print(xticks_)
    ax.set_xticks(xticks_[0],xticks_[1],rotation=90)
    ax.set_title("Boxplots for see the proportion of outliers")
    plt.show()
    st.pyplot(fig)
with st.container():
    st.markdown("Comparing the boxplot with the dataframe values, we can see that the data obey the projections")
with st.container():
    st.header("Now we will make sure that some columns have consistent values:")
    st.markdown("We will create a dictionary for each column with dataset and calculated values and make the comparison:")
with st.container():
    st.header("Making the analysis for growth rate:")
    code = '''
    growth_rate_calculated = [str(round(((1 - (df["2022 population"][i] / df["2023 population"][i])) * 100),2))+"%" for i in range(len(df))]
    growth_rate_df = [df["growth rate"][i] for i in range(len(df))]
    growth_comparisson = {float(growth_rate_df[i][0:len(growth_rate_df[i])-1]) : float(growth_rate_calculated[i][0:len(growth_rate_calculated[i])-1]) for i in range (len(df))}
    growth_comparisson       
            '''
    st.code(code, language='python')
    growth_rate_calculated = [str(round(((1 - (df["2022 population"][i] / df["2023 population"][i])) * 100),2))+"%" for i in range(len(df))]
    growth_rate_df = [df["growth rate"][i] for i in range(len(df))]
    growth_comparisson = {float(growth_rate_df[i][0:len(growth_rate_df[i])-1]) : float(growth_rate_calculated[i][0:len(growth_rate_calculated[i])-1]) for i in range (len(df))}
    growth_comparisson   
with st.container():
    st.markdown("The values are almost equivalent, so we can let the original values")
with st.container():
    st.header("Making the same analysis for density:")
    code = '''
    density_df = [df["density (km²)"][i] for i in range(len(df))]
    density_calculated = [df["2023 population"][i]/df["area (km²)"][i] for i in range(len(df))]
    density_comparison = {str(density_df[i]):density_calculated[i] for i in range(len(df))}
    density_comparison       
            '''
    st.code(code, language='python')
    density_df = [df["density (km²)"][i] for i in range(len(df))]
    density_calculated = [df["2023 population"][i]/df["area (km²)"][i] for i in range(len(df))]
    density_comparison = {str(density_df[i]):density_calculated[i] for i in range(len(df))}
    density_comparison  
with st.container():
    st.markdown("Some values are inconsistent, so we will fix them:")
with st.container():
    code = '''
    for i in range(len(df)):
        df["density (km²)"][i] = round(density_calculated[i])    
            '''
    st.code(code, language='python')
    for i in range(len(df)):
        df["density (km²)"][i] = round(density_calculated[i]) 
with st.container():
    st.header("Making the same analysis for world percentage:")
    code = '''
    world_percentage = {str(df["world percentage"][i][0:len(df["world percentage"][i])-1]):(df["2023 population"][i] / population_23 *100) for i in range(len(df))}   
            '''
    st.code(code, language='python')
    population_23 = df["2023 population"].sum()
    world_percentage = {df["world percentage"][i][0:len(df["world percentage"][i])-1]:(df["2023 population"][i] / population_23 *100) for i in range(len(df))} 
    world_percentage
with st.container():
    st.markdown("The values are almost equivalent, so we can let the original values")
with st.container():
    st.header("Now, converting the string type (object) to float64 type:")
    code = '''
    for i in range(len(df)):
        df["world percentage"][i] = float(df["world percentage"][i][0:len(df["world percentage"][i])-1])
    '''
    st.code(code, language='python')
    for i in range(len(df)):
        df["world percentage"][i] = float(df["world percentage"][i][0:len(df["world percentage"][i])-1])
with st.container():
    st.header("Now, let's see how is the dataframe after the changes")
    code = '''
    df
    '''
    st.code(code, language='python')
    df
    st.markdown("Looks like the dataframe values are OK")
with st.container():
    st.header("Making a new dataframe groupby country and continent and sum of that values:")
    code = '''
    columns = df.columns[4:13]
    df_2 = df.groupby(["country","continent"])[columns].sum().stack().reset_index()
    df_2.columns=['country','continent', 'year', 'population']
    df_2["year"] = df_2["year"].str[0:4]
    df_2["year"] = df_2["year"].astype(np.int64)
    df_2
    '''
    st.code(code, language='python')
    columns = df.columns[4:13]
    df_2 = df.groupby(["country","continent"])[columns].sum().stack().reset_index()
    df_2.columns=['country','continent', 'year', 'population']
    df_2["year"] = df_2["year"].str[0:4]
    df_2["year"] = df_2["year"].astype(np.int64)
    df_2
with st.container():
    st.header("Now, with that new dataframe, let's make an analysis of world's populace splitted by continent")
    code = '''
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(data=df_2,x="year",y="population",hue="continent",errorbar=None)
    plt.title("World's population diference per continent")
    plt.grid()
    '''
    st.code(code, language='python')
    fig,ax = plt.subplots(figsize=(12,10))
    sns.barplot(data=df_2,x="year",y="population",hue="continent",errorbar=None)
    plt.title("World's population diference per continent")
    plt.grid() 
    st.pyplot(fig)
    st.markdown("As we can see, by the plot, that the worlds populace are stopping to grow, even in Asia, the most populous continent.")
    st.markdown("Europe has the almost the same population since 70's")
    st.markdown("Africa overpassed Europe population in 2000 first decade")
    st.markdown("Oceania has the lowest population in all times")
with st.container():
    st.header("Another perspective, seeing the proportion of population splitted by continent but a % of total ")
    code = '''
    df_continent_sum = pd.pivot_table(data=df,values=columns,index="continent",aggfunc="sum").reset_index()
    fig,ax = plt.subplots(nrows=len(columns),figsize=(25,25))
    for i in range (len(columns)):
        ax[i].pie(df_continent_sum[df_continent_sum.columns[i+1]],labels=df_continent_sum["continent"],shadow=True,autopct="%1.1f%%")
        ax[i].set_title(f"{df_continent_sum.columns[i+1]}:")
    plt.tight_layout()
    plt.show()
    '''
    st.code(code, language='python')
    df_continent_sum = pd.pivot_table(data=df,values=columns,index="continent",aggfunc="sum").reset_index()
    fig,ax = plt.subplots(nrows=len(columns),figsize=(25,25))
    for i in range (len(columns)):
        ax[i].pie(df_continent_sum[df_continent_sum.columns[i+1]],labels=df_continent_sum["continent"],shadow=True,autopct="%1.1f%%")
        ax[i].set_title(f"{df_continent_sum.columns[i+1]}:")
    plt.tight_layout()
    plt.show() 
    st.pyplot(fig)
    
with st.container():
    st.header("Now, let's see the India and China growth behavior, once they're the most populous countries in Asia")
    code = '''
    df_india = df_2[df_2["country"] == "India"]
    df_china = df_2[df_2["country"] == "China"]
    df_china_india = pd.concat([df_china,df_india], join="inner")
    df_china_india
    '''
    st.code(code, language='python')
    df_india = df_2[df_2["country"] == "India"]
    df_china = df_2[df_2["country"] == "China"]
    df_china_india = pd.concat([df_china,df_india], join="inner")
    df_china_india
    st.markdown("Now, we have a new dataframe with China and Asia only information")
with st.container():
    code = '''
    sns.barplot(data=df_china_india,x="year",y="population",hue="country")
    plt.title("Comparative between China and India (The most populous countries)")
    plt.show()
    '''
    st.code(code, language='python')
    fig, ax = plt.subplots()
    sns.barplot(data=df_china_india,x="year",y="population",hue="country")
    plt.title("Comparative between China and India (The most populous countries)")
    plt.show()
    st.pyplot(fig)
    st.markdown("As we can see, The populace of India grew more than China in at least 10 last years, and in 2023 became the most populous country in world ")
with st.container():
    st.header("Now, let's make a comparison of the projections of China and India population if growth rate ramains the same:")
    st.markdown("First let's preparate the data:")
    code = '''
    years = [i+1 for i in range(2022,2042)]
    india_projection = []
    start = df["2023 population"][0]
    india_projection.append(start)
    for i in range (1,20):
        india_projection.append(round(india_projection[i-1]*(1+float(df["growth rate"][0][0:len(df["growth rate"][0])-1])/100)))
    china_projection = []
    start = df["2023 population"][1]
    china_projection.append(start)
    for i in range (1,20):
        china_projection.append(round(china_projection[i-1]*(1+float(df["growth rate"][1][0:len(df["growth rate"][1])-1])/100)))
    india_projection_dict = {years[i]:india_projection[i] for i in range(len(years))}
    china_projection_dict = {years[i]:china_projection[i] for i in range(len(years))}
    india_projection_df = pd.DataFrame.from_dict(india_projection_dict,orient="index").reset_index()
    china_projection_df = pd.DataFrame.from_dict(china_projection_dict,orient="index").reset_index()
    india_projection_df.rename(columns={"index":"year",0:"population"},inplace=True)
    china_projection_df.rename(columns={"index":"year",0:"population"},inplace=True)
    '''
    st.code(code, language='python')
    years = [i+1 for i in range(2022,2042)]
    india_projection = []
    start = df["2023 population"][0]
    india_projection.append(start)
    for i in range (1,20):
        india_projection.append(round(india_projection[i-1]*(1+float(df["growth rate"][0][0:len(df["growth rate"][0])-1])/100)))
    china_projection = []
    start = df["2023 population"][1]
    china_projection.append(start)
    for i in range (1,20):
        china_projection.append(round(china_projection[i-1]*(1+float(df["growth rate"][1][0:len(df["growth rate"][1])-1])/100)))
    india_projection_dict = {years[i]:india_projection[i] for i in range(len(years))}
    china_projection_dict = {years[i]:china_projection[i] for i in range(len(years))}
    india_projection_df = pd.DataFrame.from_dict(india_projection_dict,orient="index").reset_index()
    china_projection_df = pd.DataFrame.from_dict(china_projection_dict,orient="index").reset_index()
    india_projection_df.rename(columns={"index":"year",0:"population"},inplace=True)
    china_projection_df.rename(columns={"index":"year",0:"population"},inplace=True)
with st.container():
    st.markdown("India dataset:")
    india_projection_df
    st.markdown("China dataset:")
    china_projection_df
with st.container():
    st.markdown("Let's plot that information:")
    code = '''
    fig,ax = plt.subplots()
    sns.lineplot(data=china_projection_df,y="population",x="year",label="China",ax=ax)
    sns.lineplot(data=india_projection_df,y="population",x="year",label="India",ax=ax)
    ax.set_title("The population of China and India comparison if the growth rate remains the same")
    ax.set_xticks(np.arange(2020,2050,5),np.arange(2020,2050,5))
    ax.grid()
    plt.show()
    '''
    st.code(code, language='python')
    years = [i+1 for i in range(2022,2042)]
    india_projection = []
    start = df["2023 population"][0]
    india_projection.append(start)
    for i in range (1,20):
        india_projection.append(round(india_projection[i-1]*(1+float(df["growth rate"][0][0:len(df["growth rate"][0])-1])/100)))
    china_projection = []
    start = df["2023 population"][1]
    china_projection.append(start)
    for i in range (1,20):
        china_projection.append(round(china_projection[i-1]*(1+float(df["growth rate"][1][0:len(df["growth rate"][1])-1])/100)))
    india_projection_dict = {years[i]:india_projection[i] for i in range(len(years))}
    china_projection_dict = {years[i]:china_projection[i] for i in range(len(years))}
    india_projection_df = pd.DataFrame.from_dict(india_projection_dict,orient="index").reset_index()
    china_projection_df = pd.DataFrame.from_dict(china_projection_dict,orient="index").reset_index()
    india_projection_df.rename(columns={"index":"year",0:"population"},inplace=True)
    china_projection_df.rename(columns={"index":"year",0:"population"},inplace=True)
    fig,ax = plt.subplots()
    ax.plot(y=china_projection_df["population"],x=china_projection_df["year"])
    ax.plot(y=india_projection_df["population"],x=india_projection_df["year"])
    ax.set_title("The population of China and India comparison if the growth rate remains the same")
    ax.set_xticks(np.arange(2020,2050,5),np.arange(2020,2050,5))
    ax.grid()
    plt.show()
    st.pyplot(fig)

with st.container():
    st.header("Now, let's see how the world population grows")
    code = '''
    columns = df.columns[4:13]
    total = []
    for col in range (len(columns)):
        total.append(df[columns[col]].sum())
    years = [columns[i].split()[0] for i in range (len(columns))]
    total.reverse()
    years.reverse()
    '''
    st.code(code, language='python')
    population_23 = df["2023 population"].sum()
    columns = df.columns[4:13]
    total = []
    for col in range (len(columns)):
        total.append(df[columns[col]].sum())
    years = [columns[i].split()[0] for i in range (len(columns))]
    total.reverse()
    years.reverse()
with st.container():
    st.markdown("Creating lists with worlds sum populace and the columns(years)")
    code = '''
    worlds_populace = [df[df.columns[i]].sum() for i in range(4,13)]
    worlds_populace_columns = [int(df.columns[i].split()[0]) for i in range(4,13)]
    worlds_populace_columns.reverse()
    worlds_populace.reverse()
    '''
    st.code(code, language='python')
    worlds_populace = [df[df.columns[i]].sum() for i in range(4,13)]
    worlds_populace_columns = [int(df.columns[i].split()[0]) for i in range(4,13)]
    worlds_populace_columns.reverse()
    worlds_populace.reverse()
with st.container():
    st.header("Let's see a growth simulation of worlds populace:")
    code = '''
    world_growth_rate = 1+(worlds_populace[len(worlds_populace)-2]/worlds_populace[len(worlds_populace)-1]/100)
    for i in range (21):
        worlds_populace.append(round(worlds_populace[len(worlds_populace)-1]*world_growth_rate))
        worlds_populace_columns.append(worlds_populace_columns[len(worlds_populace_columns)-1]+1)
    worlds_predicted_populace_dict = {worlds_populace_columns[i]:worlds_populace[i] for i in range(len(worlds_populace))}
    worlds_predicted_populace_df = pd.DataFrame.from_dict(worlds_predicted_populace_dict,orient="index").reset_index()
    worlds_predicted_populace_df.columns = ["year","population"]
    worlds_predicted_populace_df
    '''
    st.code(code, language='python')
    world_growth_rate = 1+(worlds_populace[len(worlds_populace)-2]/worlds_populace[len(worlds_populace)-1]/100)
    for i in range (21):
        worlds_populace.append(round(worlds_populace[len(worlds_populace)-1]*world_growth_rate))
        worlds_populace_columns.append(worlds_populace_columns[len(worlds_populace_columns)-1]+1)
    worlds_predicted_populace_dict = {worlds_populace_columns[i]:worlds_populace[i] for i in range(len(worlds_populace))}
    worlds_predicted_populace_df = pd.DataFrame.from_dict(worlds_predicted_populace_dict,orient="index").reset_index()
    worlds_predicted_populace_df.columns = ["year","population"]
    worlds_predicted_populace_df
with st.container():
    st.markdown("Ploting the dataframe made above:")
    code = '''
    sns.barplot(data=worlds_predicted_populace_df,x="year",y="population")
    xtks = plt.xticks()
    plt.xticks(xtks[0],xtks[1],rotation=90)
    plt.title("World's population + predicted population based in the same growth rate since 2023")
    plt.show()
    '''
    st.code(code, language='python')
    fig, ax = plt.subplots()
    sns.barplot(data=worlds_predicted_populace_df,x="year",y="population")
    xtks = plt.xticks()
    plt.xticks(xtks[0],xtks[1],rotation=90)
    plt.title("World's population + predicted population based in the same growth rate since 2023")
    plt.show()
    st.pyplot(fig)
with st.container():
    st.header("Let's see the most and the least 20 populated countries")
    code = '''
    df_most_density = df[df["rank"] <= 20]
    df_least_density = df[df["rank"] >= 215].reset_index()
    df_most_density = df.sort_values(ascending=False,by="density (km²)")
    df_least_density = df.sort_values(ascending=True,by="density (km²)")
    df_least_density = df_least_density[0:20].reset_index()
    df_most_density = df_most_density[0:20].reset_index()
    least_countries = [df_least_density["country"][i] for i in range (len(df_least_density))]
    most_countries = [df_most_density["country"][i] for i in range (len(df_most_density))]
    '''
    st.code(code, language='python')
    df_most_density = df[df["rank"] <= 20]
    df_least_density = df[df["rank"] >= 215].reset_index()
    df_most_density = df.sort_values(ascending=False,by="density (km²)")
    df_least_density = df.sort_values(ascending=True,by="density (km²)")
    df_least_density = df_least_density[0:20].reset_index()
    df_most_density = df_most_density[0:20].reset_index()
    least_countries = [df_least_density["country"][i] for i in range (len(df_least_density))]
    most_countries = [df_most_density["country"][i] for i in range (len(df_most_density))]
with st.container():
    st.header("Ploting the data made above: ")
    code = '''
    fig,ax = plt.subplots(nrows=2,figsize=(8,10))
    ax[0].bar(x=df_most_density["country"],height=df_most_density["density (km²)"])
    ax[1].bar(x=df_least_density["country"],height=df_least_density["density (km²)"],color="orange")
    ax[0].set_xticks(np.arange(20),most_countries,rotation=90)
    ax[1].set_xticks(np.arange(20),least_countries,rotation=90)
    ax[0].set_title("The 20 most populated countries")
    ax[1].set_title("The 20 least populated countries")
    ax[0].set_ylabel("Density")
    ax[1].set_ylabel("Density")
    ax[0].set_xlabel("Countries")
    ax[1].set_xlabel("Countries")
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    '''
    st.code(code, language='python')
    fig,ax = plt.subplots(nrows=2,figsize=(8,10))
    ax[0].bar(x=df_most_density["country"],height=df_most_density["density (km²)"])
    ax[1].bar(x=df_least_density["country"],height=df_least_density["density (km²)"],color="orange")
    ax[0].set_xticks(np.arange(20),most_countries,rotation=90)
    ax[1].set_xticks(np.arange(20),least_countries,rotation=90)
    ax[0].set_title("The 20 most populated countries")
    ax[1].set_title("The 20 least populated countries")
    ax[0].set_ylabel("Density")
    ax[1].set_ylabel("Density")
    ax[0].set_xlabel("Countries")
    ax[1].set_xlabel("Countries")
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    st.pyplot(fig)
with st.container():
    st.header("Making a plot with the bigger and the smallest countries by the area(km²)")
    code = '''
    df_least_area = df.sort_values(by="area (km²)",ascending=True)[0:20].reset_index()
    df_most_area = df.sort_values(by="area (km²)",ascending=False)[0:20].reset_index()
    least_area_country = [df_least_area["country"][i] for i in range(len(df_least_area))]
    most_area_country = [df_most_area["country"][i] for i in range(len(df_most_area))]
    fig, ax = plt.subplots(nrows=2,figsize=(8,10))
    sns.barplot(data=df_most_area,x="country",y="area (km²)",ax=ax[0])
    sns.barplot(data=df_least_area,x="country",y="area (km²)",ax=ax[1])
    ax[0].set_xticks(np.arange(0,20),most_area_country,rotation=90)
    ax[1].set_xticks(np.arange(0,20),least_area_country,rotation=90)
    ax[0].set_title("The largest countries")
    ax[1].set_title("The tiniest countries")
    ax[1].grid()
    ax[0].grid()
    plt.tight_layout()
    plt.show()
    '''
    st.code(code, language='python')
    df_least_area = df.sort_values(by="area (km²)",ascending=True)[0:20].reset_index()
    df_most_area = df.sort_values(by="area (km²)",ascending=False)[0:20].reset_index()
    least_area_country = [df_least_area["country"][i] for i in range(len(df_least_area))]
    most_area_country = [df_most_area["country"][i] for i in range(len(df_most_area))]
    fig, ax = plt.subplots(nrows=2,figsize=(8,10))
    sns.barplot(data=df_most_area,x="country",y="area (km²)",ax=ax[0])
    sns.barplot(data=df_least_area,x="country",y="area (km²)",ax=ax[1])
    ax[0].set_xticks(np.arange(0,20),most_area_country,rotation=90)
    ax[1].set_xticks(np.arange(0,20),least_area_country,rotation=90)
    ax[0].set_title("The largest countries")
    ax[1].set_title("The tiniest countries")
    ax[1].grid()
    ax[0].grid()
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
with st.container():
    st.header("Ploting the 20 most and least populous countries ")
    code = '''
    df_2_sorted = df_2[df_2["year"] == 2023].sort_values(by="population",ascending=False).reset_index()
    df_2_sorted_most = df_2_sorted[0:20]
    df_2_sorted_least = df_2_sorted[len(df_2_sorted)-20:len(df_2_sorted)].reset_index()
    fig,ax = plt.subplots(nrows=2,figsize=(8,10))
    sns.barplot(data=df_2_sorted_most,x="country",y="population",ax=ax[0])
    sns.barplot(data=df_2_sorted_least,x="country",y="population",ax=ax[1])
    ax[0].set_xticks(np.arange(0,20),df_2_sorted_most["country"],rotation=90)
    ax[1].set_xticks(np.arange(0,20),df_2_sorted_least["country"],rotation=90)
    ax[0].set_title("The 20 country with more citizens")
    ax[1].set_title("The 20 country with less citizens")
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.show()
    '''
    st.code(code, language='python')
    df_2_sorted = df_2[df_2["year"] == 2023].sort_values(by="population",ascending=False).reset_index()
    df_2_sorted_most = df_2_sorted[0:20]
    df_2_sorted_least = df_2_sorted[len(df_2_sorted)-20:len(df_2_sorted)].reset_index()
    fig,ax = plt.subplots(nrows=2,figsize=(8,10))
    sns.barplot(data=df_2_sorted_most,x="country",y="population",ax=ax[0])
    sns.barplot(data=df_2_sorted_least,x="country",y="population",ax=ax[1])
    ax[0].set_xticks(np.arange(0,20),df_2_sorted_most["country"],rotation=90)
    ax[1].set_xticks(np.arange(0,20),df_2_sorted_least["country"],rotation=90)
    ax[0].set_title("The 20 country with more citizens")
    ax[1].set_title("The 20 country with less citizens")
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
