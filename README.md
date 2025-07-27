# Air-Quality
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pytz


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
# import our time series data from the Database
#preparation and collection of the database - skipped
df = pd.DataFrame(result).set_index("timestamp")
df.head()
# wrangle function
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    df = pd.DataFrame(results).set_index("timestamp")

    #Localize Timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    #Remove Outliers
    df = df[df["P2"]<500]

    #Resample to 1Hr Window, FFill missing Values
    df = df["P2"].resample("1H").mean().fillna(method="ffill").to_frame()

    #Add Lag Feature
    df["P2.L1"] = df["P2"].shift(1)

    #Drop NaN Rows
    df.dropna(inplace = True)
    
    return df

df =wrangle(nairobi)
print(df.shape)
df.head(10)

#Boxplot PM2.5 distribution
fig, ax = plt.subplots(figsize=(15, 6))

df["P2"].plot(kind = "box", vert = False, title = "Distribution of PM2.5 Readings", ax = ax)

#PM2.5 Time Series
fig, ax = plt.subplots(figsize=(15, 6))

df["P2"].plot(xlabel = "Time", ylabel = "PM2.5", title = "PM2.5 Time Series", ax=ax)

#Scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x = df["P2.L1"], y = df["P2"])
ax.plot([0,120], [0,120], linestyle = "--", color= "orange")
plt.xlabel = ("P2.L1")
plt.ylabel("P2")
plt.title("PM2.5 AutoCorrelation")

#BUILD MODEL
#baseline
y_pred_baseline = [y_train.mean()] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", round(y_train.mean(), 2))
print("Baseline MAE:", round(mae_baseline, 2))

model = LinearRegression()
model.fit(X_train, y_train)

#EVALUATE
training_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))
print("Training MAE:", round(training_mae, 2))
print("Test MAE:", round(test_mae, 2))

#Intercept and Coefficient
intercept = model.intercept_.round(2)
coefficient = model.coef_.round(2)

print(f"P2 = {intercept} + ({coefficient} * P2.L1)")


df_pred_test = pd.DataFrame(
    {
        "y_test": y_test,
        "y_pred": model.predict(X_test)
    }
)
df_pred_test.head()

#plotly express test prediction
fig = px.line(df_pred_test, labels = {"value": "P2"})
fig.show()
