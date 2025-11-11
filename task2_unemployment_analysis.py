''' CodeAlpha Data Science Internship
Task 2: Unemployment Analysis with Python
Author: <Harshini>
Date: November 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


data = pd.read_csv("Unemployment in India (1).csv")
print("✅ Dataset loaded successfully!\n")


print("First 5 rows of the dataset:")
print(data.head())


print("\nDataset Information:")
print(data.info())

print("\nChecking for missing values:")
print(data.isnull().sum())


data.columns = data.columns.str.strip().str.replace(" ", "_").str.replace(".", "")
print("\nCleaned Column Names:", list(data.columns))


print("\nSummary Statistics:")
print(data.describe())



data.dropna(subset=["Estimated_Unemployment_Rate(%)"], inplace=True)


num_cols = data.select_dtypes(include=[np.number]).columns
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")

plt.figure(figsize=(8, 5))
sns.histplot(data["Estimated_Unemployment_Rate(%)"], kde=True, color="steelblue")
plt.title("Distribution of Unemployment Rate (%) in India")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")
plt.show()


if "Region" in data.columns:
    plt.figure(figsize=(12, 6))
    region_avg = data.groupby("Region")["Estimated_Unemployment_Rate(%)"].mean().sort_values(ascending=False)
    sns.barplot(x=region_avg.values, y=region_avg.index, palette="viridis")
    plt.title("Average Unemployment Rate by Region in India")
    plt.xlabel("Average Unemployment Rate (%)")
    plt.ylabel("Region")
    plt.show()


if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Year"] = data["Date"].dt.year
    yearly_avg = data.groupby("Year")["Estimated_Unemployment_Rate(%)"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=yearly_avg, x="Year", y="Estimated_Unemployment_Rate(%)", marker="o", color="teal")
    plt.title("Average Unemployment Rate Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Average Unemployment Rate (%)")
    plt.show()


if "Year" in data.columns:
    pre_covid = data[data["Year"] <= 2019]["Estimated_Unemployment_Rate(%)"].mean()
    during_covid = data[data["Year"].between(2020, 2021)]["Estimated_Unemployment_Rate(%)"].mean()

    print(f"\n📈 Average Unemployment Rate Before COVID (≤2019): {pre_covid:.2f}%")
    print(f"📉 Average Unemployment Rate During COVID (2020–2021): {during_covid:.2f}%")

    plt.figure(figsize=(6, 5))
    sns.barplot(x=["Pre-COVID", "During COVID"], y=[pre_covid, during_covid], palette="Set2")
    plt.title("Impact of COVID-19 on Unemployment Rate in India")
    plt.ylabel("Average Unemployment Rate (%)")
    plt.show()


plt.figure(figsize=(8, 5))
sns.heatmap(data[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


