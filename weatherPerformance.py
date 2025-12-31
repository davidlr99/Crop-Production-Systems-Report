# https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/monthly/air_temperature_mean/
from calendar import month_abbr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pd.set_option("display.max_columns", None)
# paper: https://www.openagrar.de/receive/openagrar_mods_00092044

states = [
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

months = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}

data_names = {
    "air_temperature_mean": "Mean Temp. (C)",
    "precipitation": "Mean Precipitation (L/m²)",
    "sunshine_duration": "Sunshine duration (h)",
}

data_durations = {
    "air_temperature_mean": "from 1881-2024 (145 years)",
    "precipitation": "from 1881-2024 (145 years)",
    "sunshine_duration": "from 1951-2024 (75 years)",
}


for data_name in data_names:
    averageIncrease = 0.0
    averageRsquared = 0.0
    totalCombination = 0.0
    gains = {}
    for month_key in months:
        df = pd.read_csv(f"./weather/data/{data_name}/{month_key}.txt", sep=";")
        for state in states:
            x = df["Jahr"].values
            y = df[state].values

            m, b = np.polyfit(x, y, 1)
            y_pred = m * x + b

            gains[f"{state}-{months[month_key]}"] = m
            averageIncrease += m

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot

            averageRsquared += r2
            totalCombination += 1
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_pred_sorted = y_pred[sort_idx]

            ax = df.plot(x="Jahr", y=state, linewidth=1, label="Observed")
            ax.plot(
                x_sorted,
                y_pred_sorted,
                color="red",
                linewidth=1.5,
                label=f"Linear fit: $y={m:.3f}x+{b:.3f}$\n$R^2={r2:.3f}$",
            )

            ax.set_xlabel("Year")
            ax.set_ylabel(data_names[data_name])
            ax.set_title(
                f"{data_names[data_name]} in {months[month_key]} \nin {state} {data_durations[data_name]}"
            )

            ax.legend()
            plt.tight_layout()
            plt.savefig(
                f"weather/charts/{data_name}/{state.replace('/', '.')}_in_{months[month_key]}.png"
            )
            plt.close()
    gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    print(gains)
    print(f"Compinations looked at: {totalCombination}")
    print(f"Average {data_name} increase: {averageIncrease / totalCombination}")
    print(f"Average {data_name} R^2: {averageRsquared / totalCombination}")
