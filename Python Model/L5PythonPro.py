import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Opening files and renaming columns and fixing data types
itt_csv = 'Individual_Time_Trials.csv'
ColName = ['Year', 'Place', 'Team']
itt_df = pd.read_csv(itt_csv, names=ColName, header=None)
#itt_df = itt_df[itt_df['Year'] >= 2002]

itt_df['Year'] = itt_df['Year'].astype(int)
itt_df['Place'] = itt_df['Place'].astype(float)
itt_df['Team'] = itt_df['Team'].astype(str)

l500_csv = "Little_500_Race.csv"
l500_df = pd.read_csv(l500_csv, names=ColName, header=None)

l500_df['Year'] = l500_df['Year'].astype(int)
l500_df['Place'] = l500_df['Place'].astype(float)
l500_df['Team'] = l500_df['Team'].astype(str)

quals_csv = 'Qualifications.csv'
quals_df = pd.read_csv(quals_csv, names=ColName, header=None)

quals_df['Year'] = quals_df['Year'].astype(int)
quals_df['Place'] = quals_df['Place'].astype(float)
quals_df['Team'] = quals_df['Team'].astype(str)

teamPur_csv = "Team_Pursuit.csv"
teamPur_df = pd.read_csv(teamPur_csv, names=ColName, header=None)

teamPur_df['Year'] = teamPur_df['Year'].astype(int)
teamPur_df['Place'] = teamPur_df['Place'].astype(float)
teamPur_df['Team'] = teamPur_df['Team'].astype(str)
print("Opened Successfully and Reformatted")

# ITT data processing - Already cleaned
# Taking the average placement of each rider in a team and rounding to the nearest integer
itt_avg = itt_df.groupby(['Year', 'Team'], as_index=False)['Place'].mean().round()

# Remove teams with fewer than 4 riders
team_sizes = itt_df.groupby(['Year', 'Team']).size().reset_index(name='TeamSize')
# teams_with_four_or_more = team_sizes[team_sizes['TeamSize'] >= 4]
itt_avg_filtered = itt_avg.merge(team_sizes[['Year', 'Team']], on=['Year', 'Team'])

# Remove duplicates within a year
itt_avg_filtered = itt_avg_filtered.drop_duplicates(subset=['Year', 'Team'])

# Sort by year, average placement, and team name
itt_avg_sorted = itt_avg_filtered.sort_values(by=['Year', 'Place', 'Team'])

# Rank teams by average placement within each year, considering alphabetical order for ties
itt_avg_sorted['Rank'] = itt_avg_sorted.groupby('Year')['Place'].rank(method='first')

# Select the top 33 teams
itt_top33 = itt_avg_sorted#[itt_avg_sorted['Rank'] <= 33]
itt_merged = pd.merge(itt_df, l500_df, on=['Year','Team'], how = 'left')
itt_Processed = itt_top33[itt_top33['Team'].isin(itt_merged['Team'])]

ittPro = 'ittProcessed.csv'
itt_Processed.to_csv(ittPro, index=False)
print("Itt data processed successfully")

# Team Pursuit (cleaned) - removing all teams that aren't in the race for the same year
teamPur_merged = pd.merge(teamPur_df, l500_df, on=['Year', 'Team'], how='left')
teamPur_Processed = teamPur_df[teamPur_df['Team'].isin(teamPur_merged['Team'])]

# Save the result back to a CSV file if needed
teamPurPro = 'teamPur_filtered.csv'
teamPur_Processed.to_csv(teamPurPro, index=False)
print('Team Pursuit data processed successfully')

# Quals data I've already preprocessed
quals_merged = pd.merge(quals_df, l500_df, on=['Year', 'Team'], how='left')
quals_Processed = quals_df[quals_df['Team'].isin(quals_merged['Team'])]
print('Quals data processed successfully')


# Merging all the datasets
merged_df = pd.merge(itt_Processed, quals_Processed, on=['Year', 'Place', 'Team'], how='left')
merged_df = pd.merge(merged_df, teamPur_Processed, on=['Year', 'Place', 'Team'], how='left')
merged_l5data = pd.merge(merged_df, l500_df, on=['Year', 'Place', 'Team'], how='left')

merged_l5pro = 'little500Processed.csv'
merged_df.to_csv(merged_l5pro, index=False)
print("Merged data successfully")
# Drop the original 'Team' column from features
X = merged_l5data.drop(['Place', 'Team'], axis=1)  # Features
y = merged_l5data['Place']  # Target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Examine the coefficients of the model
coefficients = pd.Series(model.coef_, index=X.columns)
most_important_attribute = coefficients.idxmax()
print(f"The most important attribute for prediction is: {most_important_attribute}")

#Evaluating the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

#Visualizing Prediction - True Values vs Prediction Plot
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs Predictions")
plt.show()

# Fit the linear regression model using statsmodels
X_with_intercept = sm.add_constant(X_train)
model = sm.OLS(y_train, X_with_intercept).fit()

# Q-Q Plot
sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

# Residuals vs Fitted Plot
plt.scatter(model.fittedvalues, model.resid)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values Plot")
plt.show()