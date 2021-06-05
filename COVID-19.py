# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic("config", "IPCompleter.greedy=True")
get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic("matplotlib", "inline")


# %%
from datetime import datetime, timedelta

today = datetime.strftime(datetime.now(), "%m-%d-%Y")
yesterday = datetime.strftime(datetime.now() - timedelta(1), "%m-%d-%Y")
yesterday2 = datetime.strftime(datetime.now() - timedelta(1), "%m/%d/%y")


# %%
print(yesterday)


# %%
import urllib
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# NY Times
url_counties = (
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
)

url_states = (
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
)


# Johns Hopkins CSSE
jh = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/"

try:
    urllib.request(jh + today + ".csv")
    data2 = pd.read_csv(jh + today + ".csv")
except TypeError:
    data2 = pd.read_csv(jh + yesterday + ".csv")


data_c = pd.read_csv(url_counties)
data_s = pd.read_csv(url_states)


# %%
# # Latest Unemployment Claims

# latest_claims = int(master_f.claims[-1:])
# # latest_claims = master_f.claims
# # latest_claims = " ".join([str(elem) for elem in latest_claims])

# print(latest_claims)


# %%
# # Creating a Data Frame with both COV-19 cases and Unemployment Claims
# latest = pd.DataFrame(
#     {"date": [yesterday2], "cases": [int(usa_total)], "claims": [latest_claims]}
# )


# %%
# master_f = master_f.append(latest, ignore_index=True)


# %%
data_s.info()


# %%
data2.info()


# %%
# Print Counties
latest_date = data_c[-1:]
latest_date = latest_date.date
latest_date = " ".join([str(elem) for elem in latest_date])

print(latest_date)


# %%
pop_florida = pd.read_csv(
    "data/florida_county_population.csv"
)  # Scraped from https://en.wikipedia.org/wiki/List_of_counties_in_Florida


fl_latest = data_c.query("state == 'Florida'")
# fl_latest = fl_latest.tail(67)
fl_latest.drop(fl_latest[fl_latest["date"] != latest_date].index, inplace=True)
fl_latest.drop(fl_latest[fl_latest["county"] == "Unknown"].index, inplace=True)

popul_fl_counties = list(pop_florida["population"])
fl_latest["population"] = popul_fl_counties
fl_latest["mortality_rate"] = round(fl_latest["deaths"] / fl_latest["cases"], 4)
fl_latest["cases_rate_per_population"] = round(
    fl_latest["cases"] / fl_latest["population"], 4
)
fl_latest
fl_latest.to_csv("data/fl_latest.csv", index=False)
fl_latest.sort_values(by=["cases"], ascending=False).head(15).reset_index(drop=True)


# %%
pop_states = pd.read_csv(
    "data/states_populations.csv"
)  # Scraped from https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population
top_states = data_s[(data_s.date == latest_date)].copy()

top_states["population"] = list(pop_states["population"])
top_states["mortality_rate"] = round(top_states["deaths"] / top_states["cases"], 4)
top_states["cases_rate_per_population"] = round(
    top_states["cases"] / top_states["population"], 4
)
top_states.sort_values(by=["cases"], ascending=False).reset_index(drop=True)


# %%
# Counties in the US with over 4000 cases
top_counties = data_c[(data_c.cases > 4000) & (data_c.date == latest_date)]
top_counties.sort_values(by=["cases"], ascending=False).reset_index(drop=True).head(15)


# %%
total_confirmed = lambda df: df.Confirmed.sum()
total_recovered = lambda df: df.Recovered.sum()
recovered_ratio = lambda df: df.Recovered.sum() / df.Confirmed.sum()


# %%
print(f"Total Confirmed: Worldwide {total_confirmed(data2):^5,}")
print(f"Total Recovered: Worldwide {total_recovered(data2):^5,}")
print(f"Recovered/Ratio: Worldwide {recovered_ratio(data2):.4f}")


# %%
for country in ["US", "Colombia"]:
    print()
    print(
        f"Total Confirmed: {country:<15} {total_confirmed(data2.loc[(data2.Country_Region == country)]):^5,}"
    )
    print(
        f"Total Recovered: {country:<15} {total_recovered(data2.loc[(data2.Country_Region == country)]):^5,}"
    )
    print(
        f"Recovered/Ratio: {country:<15} {recovered_ratio(data2.loc[(data2.Country_Region == country)]):.4f}"
    )

# %% [markdown]
# Unsurprisingly, nor the State of Florida nor the County of Miami-Dade are reporting recovered numbers.

# %%
for province_state in ["Capital District", "Florida"]:
    print()
    print(
        f"Total Confirmed: {province_state:<15} {total_confirmed(data2.loc[(data2.Province_State == province_state)]):^5,}"
    )
    print(
        f"Total Recovered: {province_state:<15} {total_recovered(data2.loc[(data2.Province_State == province_state)]):^5,}"
    )
    print(
        f"Recovered/Ratio: {province_state:<15} {recovered_ratio(data2.loc[(data2.Province_State == province_state)]):.4f}"
    )


# %%
top_countries_v2 = pd.DataFrame()
countries = []
counts = []


for c in set(data2.Country_Region.unique()):
    totals = data2.query("Country_Region == @c").Confirmed.sum()
    # print(f"{c},{totals}")
    countries.append(c)
    counts.append(totals)

top_countries_v2["country"] = countries
top_countries_v2["total_confirmed"] = counts


# %%
top_countries_v2.sort_values(by=["total_confirmed"], ascending=False).reset_index(
    drop=True
).head(15)


# %%
# Countries with over 10000 confirmed COV-19 cases
top_countries = data2[(data2.Confirmed > 10000)]
top_countries.sort_values(by=["Confirmed"], ascending=False).reset_index(
    drop=True
).head(15)


# %%
# Total cases in Colombia
colombia = data2.query("Country_Region == 'Colombia'").reset_index(drop=True)
colombia.sort_values(by=["Confirmed"], ascending=False).reset_index(drop=True)


# %%
# Total cases in Venezuela
vzla = data2.query("Country_Region == 'Venezuela'").reset_index(drop=True)
vzla


# %%
# Total cases in Miami
miami = data2.query("FIPS == 12086").reset_index(drop=True)
miami


# %%
miami_timeline = data_c.query('county == "Miami-Dade"').reset_index(drop=True)
miami_timeline
# miami_timeline["k"] = round(top_states["deaths"] / top_states["cases"], 4)


# %%
# Total cases in Florida
florida = data2.loc[(data2.Province_State == "Florida"), "Confirmed"].sum()
print(f"Total Cases in Florida: {florida: ,}")


# %%
s_florida = data2[
    ((data2.FIPS == 12011) | (data2.FIPS == 12086) | (data2.FIPS == 12099))
].reset_index(drop=True)
s_florida


# %%
# Count of total cases in South Florida
sflorida_sum = s_florida.loc[s_florida["Confirmed"] > 0, "Confirmed"].sum()
print(f"South Florida (Miami-Dade, Broward, Palm Beach): {sflorida_sum: ,}")


# %%
# Proportion of South Florida cases relative to Florida
print("South Florida Proportion:", round(sflorida_sum / florida, 4))


# %%
# Total cases in the United States
usa = data2.loc[(data2.Country_Region == "US"), "Confirmed"].sum()
print(f"Total Cases in the US: {usa: ,}")


# %%
# Total cases Worldwide
worldwide = data2["Confirmed"].sum()
print(f"Total Cases Worldwide: {worldwide: ,}")

# %% [markdown]
# ### Consolidating cases per date

# %%
dates = data_s["date"].unique()

rows = len(dates)


dates_dict = {"date": list(dates)}

dates_df = pd.DataFrame(data=dates_dict)

cases_list = []

for i in dates:
    cases = data_s.loc[data_s["date"] == i, "cases"].sum()
    cases_list.append(cases)
    # print(i, cases)

cases_dict = {"cases": list(cases_list)}


cases_df = pd.DataFrame(data=cases_dict)


master = dates_df.join(cases_df)


# %%
# # Cases in Florida Consolidation

dates = data_s["date"].unique()

rows = len(dates)

dates_dict = {"date": list(dates)}

dates_df = pd.DataFrame(data=dates_dict)

cases_list_fl = []

for i in dates:
    cases = data_s.loc[
        ((data_s["date"] == i) & (data_s["state"] == "Florida")), "cases"
    ].sum()
    cases_list_fl.append(cases)


cases_dict_fl = {"cases": list(cases_list_fl)}


cases_df_fl = pd.DataFrame(data=cases_dict_fl)


master_fl = dates_df.join(cases_df_fl)
master_fl["date"] = pd.to_datetime(master_fl["date"])
# master_fl.tail(48)


# %%
# # Cases in Miami-Dade County Consolidation

dates = data_s["date"].unique()

rows = len(dates)

dates_dict = {"date": list(dates)}

dates_df = pd.DataFrame(data=dates_dict)

cases_list_mdc = []

for i in dates:
    cases = data_c.loc[
        ((data_c["date"] == i) & (data_c["fips"] == 12086)), "cases"
    ].sum()
    cases_list_mdc.append(cases)

cases_dict_mdc = {"cases": list(cases_list_mdc)}


cases_df_mdc = pd.DataFrame(data=cases_dict_mdc)

master_mdc = dates_df.join(cases_df_mdc)


# %%
import datetime as dt

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

plt.rcParams["figure.dpi"] = 140

# Plot the Data
plt.figure(figsize=(10, 8))
plt.style.use("fivethirtyeight")

now = dt.datetime.now() - dt.timedelta(days=1)
start = now - dt.timedelta(days=len(dates))  # + dt.timedelta(days=1)
start_fl = now - dt.timedelta(days=50)
days = mdates.drange(start, now, dt.timedelta(days=1))
days_fl = mdates.drange(start_fl, now, dt.timedelta(days=1))


# formatter = FuncFormatter(log_10_product)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(formatter))
plt.yscale("log")

# formatter = FuncFormatter(log_10_product)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(formatter))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))

plt.plot(days, master["cases"], label="Total US Cases")
plt.plot(days, master_fl["cases"], label="Total Cases in Florida")
numbers = ["1", "10", "100", "1,000", "10,000", "100,000", "1,000,000", "10,000,000"]
plt.yticks(ticks=(1, 10, 100, 1000, 10000, 100000, 1000000, 10000000), labels=numbers)
plt.gcf().autofmt_xdate()

plt.title("Cases of COVID-19 in the United States")
plt.xlabel(r"Dates", fontsize=12)
plt.ylabel(r"Confirmed Cases", fontsize=12)


# plt.plot("date", "claims", label="Unemployment", data=master_f)
plt.legend()
plt.savefig("figures/usa_cases.svg", bbox_inches="tight")
plt.show()


# %%
us = pd.read_csv(
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv",
    parse_dates=True,
    index_col=0,
)
us["new_cases"] = us[["cases"]].diff()


# %%
us


# %%
# us.plot()


# %%
master_fl["new_cases"] = master_fl[["cases"]].diff()


# %%
master_fl = master_fl.set_index("date")


# %%
plt.figure(figsize=(10, 8))
plt.plot(us["new_cases"], label="United States")
plt.plot(master_fl["new_cases"], label="Florida")
plt.title("New Cases of COVID-19")
plt.yscale("log")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
plt.yticks(ticks=(1, 10, 100, 1000, 10000, 100000, 1000000, 10000000), labels=numbers)
plt.legend()
plt.savefig("figures/new_cases.svg", bbox_inches="tight")
plt.show()


# %%
florida


# %%
florida = data_c.query("state == 'Florida'").sort_values(by=["county", "date"])
florida["new_cases"] = florida[["cases"]].diff()

florida.query("date == '2020-06-14'").sort_values(by="new_cases", ascending=False)


# %%
import datetime as dt

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

# Plot the Data

plt.figure(figsize=(10, 8))

now = dt.datetime.now() - dt.timedelta(days=1)
start = now - dt.timedelta(days=len(dates))  # + dt.timedelta(days=1)
start_fl = dt.datetime(2020, 2, 29)
days = mdates.drange(start, now, dt.timedelta(days=1))
days_fl = mdates.drange(start_fl, now, dt.timedelta(days=1))

# formatter = FuncFormatter(log_10_product)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(formatter))
plt.yscale("log")
# formatter = FuncFormatter(log_10_product)
# plt.gca().yaxis.set_major_formatter(ScalarFormatter(formatter))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))

plt.plot(
    days_fl, master_mdc["cases"].tail(len(days_fl)), label="Total Miami-Dade Cases"
)
plt.plot(days_fl, master_fl["cases"].tail(len(days_fl)), label="Total Florida Cases")
# plt.plot(days, master["cases"], label="Total US Cases")
plt.gcf().autofmt_xdate()

plt.title("Cases of COVID-19 in Florida")
plt.xlabel(r"Dates", fontsize=12)
plt.ylabel(r"Confirmed Cases", fontsize=12)
plt.yticks(ticks=(1, 10, 100, 1000, 10000, 100000, 1000000, 10000000), labels=numbers)


# plt.plot("date", "claims", label="Unemployment", data=master_f)
plt.legend()
plt.savefig("figures/fl_cases.svg", bbox_inches="tight")
plt.show()


# %%
florida = data_c.query("state == 'Florida'").reset_index(drop=True)


# %%
florida = florida.sort_values(by=["county", "date"]).reset_index(drop=True)


# %%
florida["new_cases"] = florida.cases.diff()


# %%
june_fl = florida.query("date >= '2020-06-01'")


# %%
# june_fl[["date", "county", "new_cases"]].sort_values(
#     by=["new_cases", "date"], ascending=False
# ).set_index("county").groupby("county").plot()


# %%
import datetime

import pandas_datareader as pdr

start = datetime.datetime(2020, 3, 21)

icsa = pdr.get_data_fred("ICSA", start, today)
# icsa = pdr.DataReader("ICSA", "fred", start, today)

icsa


# %%
pdr.get_data_fred("ICSA", "07/01/2020", today).plot(
    title="Initial Weekly Unemployment Claims\n in the United States"
)
