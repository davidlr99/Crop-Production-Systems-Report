import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
# paper: https://www.openagrar.de/receive/openagrar_mods_00092044
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


def makeStateXCropYieldCharts(df_yield):
    gains = {}
    averageIncrease = 0.0
    averageRsquared = 0.0
    totalCombination = 0.0

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

            if n_years < 43 or has_any_nan:
                continue

            x = df_mean["year"].values
            y = df_mean["value"].values

            m, b = np.polyfit(x, y, 1)
            y_pred = m * x + b
            gains[f"{crop}-{state}"] = m

            averageIncrease += m
            totalCombination += 1

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot

            averageRsquared += r2

            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_pred_sorted = y_pred[sort_idx]

            ax = df_mean.plot(x="year", y="value", linewidth=1, label="Observed")
            ax.plot(
                x_sorted,
                y_pred_sorted,
                color="red",
                linewidth=1.5,
                label=f"Linear fit: $y={m:.3f}x+{b:.3f}$\n$R^2={r2:.3f}$",
            )

            ax.set_xlabel("Year")
            ax.set_ylabel("Yield (t/ha)")
            ax.set_title(f"{crop}-Yield in {state}")

            ax.legend()
            plt.tight_layout()
            plt.savefig(f"generated_charts/{crop}-{state}.png")
            plt.close()

    gains = sorted(gains.items(), key=lambda x: x[1], reverse=True)
    print(gains)
    print(f"Looked at {len(gains)} crop-state combinations (data from 1979-2021)")
    print(f"Average gain: {averageIncrease / totalCombination}t/ha increase per y")
    print(f"Average R^2: {averageRsquared / totalCombination} ")


makeStateXCropYieldCharts(df_yield)

##############

# # 1) Filter
# df_pot = df_yield[(df_yield["var"] == "potat_tot") & (df_yield["measure"] == "yield")]
# print(df_pot.head(50))

# # 2) Optional: Gruppe falls du mehrere Einträge pro Jahr+Bundesland hast
# df_mean = df_pot.groupby(["nuts_id", "year"])["value"].mean().reset_index()
# print(df_mean.head(50))

# # 3) Pivot: index=year, columns=nuts_id, values=mean value
# wide = df_mean.pivot(index="year", columns="nuts_id", values="value")
# print("wide")
# print(wide)
# # 4) Plot (pandas nutzt matplotlib)
# plt.figure(figsize=(12, 6))
# wide.plot(ax=plt.gca(), linewidth=1)
# plt.xlabel("Year")
# plt.ylabel("Yield (t/ha)")  # passe Beschriftung an
# plt.title("Kartoffel-Ertrag (potat_tot) pro Jahr — pro Bundesland")
# plt.legend(
#     title="Bundesland", bbox_to_anchor=(1.02, 1), loc="upper left"
# )  # Legende rechts
# plt.tight_layout()
# plt.show()


# ##############

# print(df_yield.head(50))


# res = df_yield.groupby(["nuts_id", "year", "var", "measure"])["value"].mean()
# print(res.head(50))
# print(res.loc[["Baden-Württemberg"]])
# years = (
#     res.loc["Baden-Württemberg"].index.get_level_values("year").unique().sort_values()
# )
# years_list = years.tolist()
# print(years)


# for idx, val in res.items():
#     # idx ist ein Tuple: (nuts_id, year, var, measure)
#     print(idx, val)
