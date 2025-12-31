import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
df_yield = pd.read_csv("./Final_data.csv", sep=",")


print(list(df_yield.columns.values))
nuts_map = {
    "DE1": "Baden-Württemberg",
    "DE2": "Bayern",
    "DE3": "Berlin",
    "DE4": "Brandenburg",
    "DE5": "Bremen",
    "DE6": "Hamburg",
    "DE7": "Hessen",
    "DE8": "Mecklenburg-Vorpommern",
    "DE9": "Niedersachsen",
    "DEA": "Nordrhein-Westfalen",
    "DEB": "Rheinland-Pfalz",
    "DEC": "Saarland",
    "DED": "Sachsen",
    "DEE": "Sachsen-Anhalt",
    "DEF": "Schleswig-Holstein",
    "DEG": "Thüringen",
}


states_nuts_to_weather_state_map = {
    "Berlin": "Brandenburg/Berlin",
    "Brandenburg": "Brandenburg",
    "Baden-Württemberg": "Baden-Wuerttemberg",
    "Bayern": "Bayern",
    "Hessen": "Hessen",
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "Niedersachsen": "Niedersachsen",
    "Bremen": "Niedersachsen/Hamburg/Bremen",
    "Nordrhein-Westfalen": "Nordrhein-Westfalen",
    "Rheinland-Pfalz": "Rheinland-Pfalz",
    "Schleswig-Holstein": "Schleswig-Holstein",
    "Saarland": "Saarland",
    "Sachsen": "Sachsen",
    "Sachsen-Anhalt": "Sachsen-Anhalt",
    "x": "Thueringen/Sachsen-Anhalt",
    "Thüringen": "Thueringen",
}

weather_states = [
    "Brandenburg/Berlin",
    "Brandenburg",
    "Baden-Wuerttemberg",
    "Bayern",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Niedersachsen/Hamburg/Bremen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Schleswig-Holstein",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Thueringen/Sachsen-Anhalt",
    "Thueringen",
]

crops_map = {
    "sb": "Spring barley",
    "wb": "Winter barley",
    "grain_maize": "Grain maize",
    "silage_maize": "Silage maize",
    "oats": "Oats",
    "potat_tot": "Potatoes",
    "wrape": "Winter rape",
    "rye": "Rye",
    "sugarbeet": "Sugarbeet",
    "triticale": "Triticale",
    "ww": "Winter wheat",
}

growing_dates = {
    "Spring barley": [
        3,
        7,  # 120 days vegetative phase
    ],  # https://www.agrarheute.com/pflanze/getreide/ratgeber-5-tipps-aussaat-sommergerste-444400
    "Winter barley": [
        9,
        7,
    ],  # https://www.kws.com/de/de/beratung/aussaat/gerste/, https://www.effizientduengen.de/getreidekulturen/wintergerste/
    "Grain maize": [
        4,  # https://www.baywa.de/de/i/beratung/mais/aussaat/maisaussaat/
        11,  # https://knalle.berlin/blogs/pop-wissen/wie-aus-maispflanzen-exzellente-maiskoerner-werden
    ],
    "Silage maize": [  # the same?
        4,  # https://www.baywa.de/de/i/beratung/mais/aussaat/maisaussaat/
        11,  # https://knalle.berlin/blogs/pop-wissen/wie-aus-maispflanzen-exzellente-maiskoerner-werden
    ],
    "Oats": [  # mostly summer Oats in Germany
        3,  # https://www.brueggen.com/de/anbauempfehlung-fur-hafer-zur-ernte-2025/
        7,  # https://www.hauptsaaten.de/fileadmin/hauptsaaten/files/2021_Haferfibel.pdf
    ],
    "Potatoes": [
        3,  # https://www.krautundrueben.de/kartoffeln-anbauen-und-ernten-das-sollten-sie-beachten-2505
        10,
    ],
    "Winter rape": [
        8,  # https://www.kws.com/de/de/beratung/ernte/winterraps/
        7,
    ],
    "Rye": [
        9,  # https://de.wikipedia.org/wiki/Roggen
        7,
    ],
    "Sugarbeet": [
        3,  # https://www.zuckerverbaende.de/anbau-und-verarbeitung/ruebenanbau/
        11,
    ],
    "Triticale": [
        9,  # https://hortica.de/getreide-ernte-reihenfolge/
        7,
    ],
    "Winter wheat": [
        10,  # https://www.strickhof.ch/publikationen/merkblatt-winterweizen/
        7,
    ],
}

# weakness: growing periods have changed over time. Also different varierties have different growing dates ofc


def map_states(x):
    for key in nuts_map:
        if key in x:
            return nuts_map[key]


def map_crops(x):
    for key in crops_map:
        if key == x:
            return crops_map[key]
    return x


# map states
df_yield["nuts_id"] = df_yield["nuts_id"].apply(map_states)

# map crops
df_yield["var"] = df_yield["var"].apply(map_crops)


# Resolution at state level
# stat_indivual_yield_points = df_yield[(df_yield["measure"] == "yield")]
# print(len(stat_indivual_yield_points))

# stat_indivual_yield_points = (
#     stat_indivual_yield_points.groupby(["nuts_id", "year", "var"])["value"]
#     .mean()
#     .reset_index()
# )
# # print(stat_indivual_yield_points)

# print(len(stat_indivual_yield_points.index))


print(df_yield.head(50))


df_yield["gperiod_air_temperature"] = np.nan
df_yield["gperiod_precipitation"] = np.nan
df_yield["gperiod_sunshine_duration"] = np.nan


months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

data_names = {
    "air_temperature_mean": "Mean Temp. (C)",
    "precipitation": "Mean Precipitation (L/m²)",
    "sunshine_duration": "Sunshine duration (h)",
}


def readWeatherData():
    final_data = {}
    for data_name in data_names:
        if data_name not in final_data:
            final_data[data_name] = {}

        for month in months:
            month_key = ""
            if month < 10:
                month_key = f"0{month}"
            else:
                month_key = f"{month}"

            df = pd.read_csv(f"./weather/data/{data_name}/{month_key}.txt", sep=";")
            for state in weather_states:
                if state not in final_data[data_name]:
                    final_data[data_name][state] = {}

                if month not in final_data[data_name][state]:
                    final_data[data_name][state][month] = {}

                year_values = df["Jahr"].values
                state_values = df[state].values

                i = 0
                for year in year_values:
                    if year not in final_data[data_name][state]:
                        final_data[data_name][state][year] = {}

                    final_data[data_name][state][year][month] = state_values[i]
                    i += 1

    return final_data


weather_data = readWeatherData()
print(weather_data["sunshine_duration"]["Bayern"][2000][1])


def prepareYieldData(df_yield, weather_data):
    input_data = []
    for state in nuts_map.values():
        for crop in crops_map.values():
            print(f"Looking at {crop} in {state}")
            df_state_crop_yield = df_yield[
                (df_yield["var"] == crop)
                & (df_yield["measure"] == "yield")
                & (df_yield["nuts_id"] == state)
            ]
            df_mean = (
                df_state_crop_yield.groupby(["nuts_id", "year"])["value"]
                .mean()
                .reset_index()
            )

            df_mean["year"] = df_mean["year"].astype(int)

            n_years = len(df_mean.index)
            has_any_nan = df_mean.isna().values.any()

            if (
                n_years < 43 or has_any_nan
            ):  # only take crop x state combinations that have all years and no missing years
                continue

            x = df_mean["year"].values
            y = df_mean["value"].values

            i = 0
            for year in x:
                growing_months = growing_dates[crop]
                virtual_start_year = year
                total_months = growing_months[1] - growing_months[0]
                if growing_months[1] < growing_months[0]:
                    virtual_start_year = year - 1
                    total_months = 12 + total_months

                res = {}
                print(f"In state {state}, looking at crop {crop} at year: {year}")
                print(
                    f"Total months: {total_months} ({growing_months[1]} - {growing_months[0]} => start: {virtual_start_year}"
                )

                for x in range(0, 12):
                    month = growing_months[0] + x
                    add_to_year = 0
                    if month > 12:
                        month = month - 12
                        add_to_year = 1

                    for data_name in data_names:
                        if f"{data_name}_{month}" not in res:
                            res[f"{data_name}_{month}"] = weather_data[data_name][
                                states_nuts_to_weather_state_map[state]
                            ][virtual_start_year + add_to_year][month]

                for x in range(0, total_months):
                    month = growing_months[0] + x
                    add_to_year = 0
                    if month > 12:
                        month = month - 12
                        add_to_year = 1

                    print(
                        f"current month: {month}, year: {virtual_start_year + add_to_year}"
                    )

                    for data_name in data_names:
                        if data_name not in res:
                            res[data_name] = 0.0

                        res[data_name] += (
                            weather_data[data_name][
                                states_nuts_to_weather_state_map[state]
                            ][virtual_start_year + add_to_year][month]
                            / total_months
                        )

                if i > 0:
                    res["crop"] = crop
                    res["state"] = state
                    res["year"] = year
                    res["yield"] = y[i]
                    res["yield_previous"] = y[i - 1]
                    print(res)
                    input_data.append(res)
                i += 1
    return input_data


input_data = prepareYieldData(df_yield, weather_data)
print(f"Training data length: {len(input_data)}")


df = pd.DataFrame(input_data)

df["yield_change"] = np.where(df["yield"] > (df["yield_previous"] + 0.1854), 1, 0)

count_better = df["yield_change"].value_counts().get(1, 0)  # Anzahl 1
count_worse = df["yield_change"].value_counts().get(0, 0)  # Anzahl 0

total_count = count_better + count_worse

percent_better = (count_better / total_count) * 100 if total_count > 0 else 0
percent_worse = (count_worse / total_count) * 100 if total_count > 0 else 0

print(f"Anzahl besser: {count_better} ({percent_better:.2f}%)")
print(f"Anzahl schlechter: {count_worse} ({percent_worse:.2f}%)")


base_features = [
    "air_temperature_mean",
    "precipitation",
    "sunshine_duration",
]
months = range(1, 5)
features_to_add = []

# remove for loop for model 1
for month in months:
    print(f"adding {month}")
    features_to_add.append(f"air_temperature_mean_{month}")
    features_to_add.append(f"precipitation_{month}")


f = base_features + features_to_add
features = df[f]

one_hot = pd.get_dummies(df["crop"])
# remove one hot encoding for model 1
features = pd.concat([features, one_hot], axis=1)
target = df["yield_change"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Report:")
print(report)


importances = model.feature_importances_
feature_names = features.columns

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(importances)), importances[indices], align="center")
plt.yticks(range(len(importances)), feature_names[indices])
plt.xlabel("Relative Feature Importance")
plt.show()
