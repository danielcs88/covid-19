# -*- coding: utf-8 -*-
# %% [markdown]
"""
# Florida $R_0$ per county
"""

# %% [markdown]
"""
Implementation using [Kevin Systrom's model](https://github.com/k-sys/covid-19)
and using the adaptation by [Ashutosh Sanzgiri per
county](https://github.com/k-sys/covid-19/blob/e95ae71f1eea827baffce2d308f767634951f9e3/Realtime_R0_by_county.ipynb)
to analyze the case for Florida and its counties.
"""


# %%
from IPython import get_ipython

get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")

# %%
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num, num2date

# from matplotlib.patches import Patch
from scipy import stats as sps
from scipy.interpolate import interp1d

# FILTERED_REGIONS = [
# "Virgin Islands",
# "American Samoa",
# "Northern Mariana Islands",
# "Guam",
# "Puerto Rico",
# ]

# FILTERED_REGION_CODES = [
# "AS", "GU", "PR", "VI", "MP"
# ]


# %%
from cycler import cycler

custom = cycler(
    "color",
    [
        "#B3220F",
        "#F16E53",
        "#FFC475",
        "#006F98",
        "#1ABBEF",
        "#7FD2FD",
        "#153D53",
        "#0F9197",
    ],
)


plt.rc("axes", prop_cycle=custom)
plt.rcParams["figure.dpi"] = 140


# %%


def highest_density_interval(pmf, p=0.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if isinstance(pmf, pd.DataFrame):
        return pd.DataFrame(
            [highest_density_interval(pmf[col], p=p) for col in pmf], index=pmf.columns
        )

    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i + 1 :]):
            if (high_value - value > p) and (not best or j < best[1] - best[0]):
                best = (i, i + j + 1)
                break

    low = pmf.index[best[0]]
    high = pmf.index[best[1]]
    return pd.Series([low, high], index=[f"Low_{p*100:.0f}", f"High_{p*100:.0f}"])


# %% [markdown]
"""
## Real-World Application to US Data (by counties)

### Setup

Load US state case data from The New York Times: County Data
"""


# %%
url_counties = (
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
)
counties = pd.read_csv(url_counties)

# Print Counties
latest_date = counties[-1:]
latest_date = latest_date.date
latest_date = " ".join(str(elem) for elem in latest_date)

print(latest_date)


# %%
state_list = sorted(set(counties.state.unique()) - set(FILTERED_REGIONS))
len(state_list)  # Include District of Columbia


# %%
w = widgets.Dropdown(
    options=state_list, description="Select state:", value="Florida", disabled=False,
)
display(w)

# %% [markdown]
"""
### Filters

* Selected state
* Remove counties listed as "Unknown"
* Remove rows with less than `county_case_filter` cases
* Remove counties with less than `county_row_filter` rows after smoothing
"""

# %%
county_case_filter = 10
county_row_filter = 5


# %%
selected_state = w.value
counties = counties[counties.state == selected_state].copy()
counties = counties[counties.county != "Unknown"].copy()
counties = counties[counties.cases >= county_case_filter].copy()
counties.shape


# %%
counties.tail()
print(len(counties))


# %%
counties = counties[["date", "county", "cases"]].copy()
counties["date"] = pd.to_datetime(counties["date"])
counties = counties.set_index(["county", "date"]).squeeze().sort_index()


# %%
counties


# %%
counties_g = (
    counties.groupby(["county"]).count().reset_index().rename({"cases": "rows"}, axis=1)
)
counties_g


# %%
county_list = counties_g[counties_g.rows >= county_row_filter]["county"].tolist()
print(len(county_list))

md = county_list.index("Miami-Dade")


# %%
county_name = county_list[md]


def prepare_cases(cases, cutoff=1):
    new_cases = cases.diff()

    smoothed = (
        new_cases.rolling(7, win_type="gaussian", min_periods=1, center=True)
        .mean(std=3)
        .round()
    )

    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


cases = counties.xs(county_name).rename(f"{county_name} cases")

original, smoothed = prepare_cases(cases)

plt.rcParams["figure.dpi"] = 140

original.plot(
    title=f"{county_name} | New Cases per Day as of [{latest_date}]",
    c="k",
    linestyle=":",
    alpha=0.5,
    label="Actual",
    legend=True,
    figsize=(500 / 72, 300 / 72),
)

ax = smoothed.plot(label="7-Day Average", legend=True)

ax.get_figure().set_facecolor("w")

plt.savefig("figures/mdc_cases.svg", bbox_inches="tight")


# %%
# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316

GAMMA = 1 / 7

# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)


def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam), index=r_t_range, columns=sr.index[1:]
    )

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range, scale=sigma).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range, columns=sr.index, data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator + 1)

    return posteriors, log_likelihood


# Note that we're fixing sigma to a value just for the example
posteriors, log_likelihood = get_posteriors(smoothed, sigma=0.25)


# %%
# ax = posteriors.plot(
#     title=f"{county_name} | Daily Posterior for $R_t$ as of {latest_date}",
#     legend=False,
#     lw=1,
#     c="k",
#     alpha=0.3,
#     xlim=(0.4, 6),
# )

# ax.set_xlabel("$R_t$");


# %%
# # Note that this takes a while to execute - it's not the most efficient algorithm
# hdis = highest_density_interval(posteriors, p=0.9)

# most_likely = posteriors.idxmax().rename("ML")

# # Look into why you shift -1
# result = pd.concat([most_likely, hdis], axis=1)

# result.tail()


# %%
sigmas = np.linspace(1 / 20, 1, 20)

targets = counties.index.get_level_values("county").isin(county_list)
counties_to_process = counties.loc[targets]

results = {}
failed_counties = []
skipped_counties = []

for county_name, cases in counties_to_process.groupby(level="county"):

    print(county_name)
    new, smoothed = prepare_cases(cases, cutoff=1)

    if len(smoothed) < 5:
        skipped_counties.append(county_name)
        continue

    result = {"posteriors": [], "log_likelihoods": []}

    try:
        for sigma in sigmas:
            posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
            result["posteriors"].append(posteriors)
            result["log_likelihoods"].append(log_likelihood)
        # Store all results keyed off of state name
        results[county_name] = result
        # clear_output(wait=True)
    except:
        failed_counties.append(county_name)
        print(f"Posteriors failed for {county_name}")

print(f"Posteriors failed for {len(failed_counties)} counties: {failed_counties}")
print(f"Skipped {len(skipped_counties)} counties: {skipped_counties}")
print(f"Continuing with {len(results)} counties / {len(county_list)}")
print("Done.")


# %%
# Each index of this array holds the total of the log likelihoods for
# the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for county_name, result in results.items():
    total_log_likelihoods += result["log_likelihoods"]

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()
# print(max_likelihood_index)

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]

# Plot it
fig, ax = plt.subplots()
ax.set_title(f"Maximum Likelihood value for $\sigma$ = {sigma:.2f}")
ax.plot(sigmas, total_log_likelihoods)
ax.axvline(sigma, color="k", linestyle=":")

# %% [markdown]
"""
### Compile Final Results

Given that we've selected the optimal $\sigma$, let's grab the precalculated
posterior corresponding to that value of $\sigma$ for each state. Let's also
calculate the 90% and 50% highest density intervals (this takes a little while)
and also the most likely value.
"""

# %%
final_results = None
hdi_error_list = []

for county_name, result in results.items():
    print(county_name)
    try:
        posteriors = result["posteriors"][max_likelihood_index]
        hdis_90 = highest_density_interval(posteriors, p=0.9)
        hdis_50 = highest_density_interval(posteriors, p=0.5)
        most_likely = posteriors.idxmax().rename("ML")
        result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
        if final_results is None:
            final_results = result
        else:
            final_results = pd.concat([final_results, result])
        clear_output(wait=True)
    except:
        print(f"HDI failed for {county_name}")
        hdi_error_list.append(county_name)
print(f"HDI error list: {hdi_error_list}")
print("Done.")

# %% [markdown]
# ### Plot All Counties meeting criteria

# %%
def plot_rt(result, ax, county_name):

    ax.set_title(f"{county_name}")

    # Colors
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(
        np.r_[np.linspace(BELOW, MIDDLE, 25), np.linspace(MIDDLE, ABOVE, 25)]
    )
    color_mapped = lambda y: np.clip(y, 0.5, 1.5) - 0.5

    index = result["ML"].index.get_level_values("date")
    values = result["ML"].values

    # Plot dots and line
    ax.plot(index, values, c="k", zorder=1, alpha=0.25)
    ax.scatter(
        index,
        values,
        s=40,
        lw=0.5,
        c=cmap(color_mapped(values)),
        edgecolors="k",
        zorder=2,
    )

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(
        date2num(index),
        result["Low_90"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )

    highfn = interp1d(
        date2num(index),
        result["High_90"].values,
        bounds_error=False,
        fill_value="extrapolate",
    )

    extended = pd.date_range(
        start=pd.Timestamp("2020-03-01"), end=index[-1] + pd.Timedelta(days=1)
    )

    ax.fill_between(
        extended,
        lowfn(date2num(extended)),
        highfn(date2num(extended)),
        color="k",
        alpha=0.1,
        lw=0,
        zorder=3,
    )

    ax.axhline(1.0, c="k", lw=1, label="$R_t=1.0$", alpha=0.25)

    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(0)
    ax.grid(which="major", axis="y", c="k", alpha=0.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0, 5.0)
    ax.set_xlim(
        pd.Timestamp("2020-03-01"),
        result.index.get_level_values("date")[-1] + pd.Timedelta(days=1),
    )
    fig.set_facecolor("w")


# %%
fr_rt = final_results[["ML", "Low_90", "High_90"]]


# %%
final_results


# %%
get_ipython().run_line_magic("time", "")
final_counties = list(fr_rt.index.get_level_values("county").unique())


# %%
c = widgets.Dropdown(
    options=final_counties,
    description="Select county:",
    value="Miami-Dade",
    disabled=False,
)
display(c)


# %%
county_name = c.value


# %%
fr_rt = (
    fr_rt.loc[(fr_rt.index.get_level_values("county") == county_name)]
    .reset_index()
    .set_index("date")
)


# %%
result = fr_rt.drop(columns=["county"])


# %%
fig, ax = plt.subplots(figsize=(600 / 72, 400 / 72))

plot_rt(result, ax, county_name)
ax.set_title(f"{county_name}: Real-time $R_t$ | {latest_date}")
# ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
plt.savefig("figures/mdc_rt.svg", bbox_inches="tight")


# %%
ncols = 3
nrows = int(np.ceil(len(final_results.groupby("county")) / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 3), dpi=120)

for i, (county_name, result) in enumerate(final_results.groupby("county")):
    plot_rt(result, axes.flat[i], county_name)

fig.tight_layout()
fig.set_facecolor("w")
plt.savefig("figures/fl_counties_rt.svg")

# %% [markdown]
# ### Export Data to CSV

# %%
# Uncomment the following line if you'd like to export the data
final_results.to_csv(f"data/rt_{selected_state}.csv")

# %% [markdown]
# ### Standings

# %%
FULL_COLOR = [0.7, 0.7, 0.7]
NONE_COLOR = [179 / 255, 35 / 255, 14 / 255]
PARTIAL_COLOR = [0.5, 0.5, 0.5]
ERROR_BAR_COLOR = [0.3, 0.3, 0.3]


# %%
final_results


# %%
filtered = final_results.index.get_level_values(0).isin(FILTERED_REGIONS)
mr = final_results.loc[~filtered].groupby(level=0)[["ML", "High_90", "Low_90"]].last()


def plot_standings(mr, figsize=None, title="Most Likely Recent $R_t$ by County"):
    if not figsize:
        figsize = ((15.9 / 50) * len(mr) + 0.1, 4.6)

    fig, ax = plt.subplots(figsize=figsize, dpi=120)

    ax.set_title(title)
    err = mr[["Low_90", "High_90"]].sub(mr["ML"], axis=0).abs()
    bars = ax.bar(
        mr.index,
        mr["ML"],
        width=0.825,
        color=FULL_COLOR,
        ecolor=ERROR_BAR_COLOR,
        capsize=2,
        error_kw={"alpha": 0.5, "lw": 1},
        yerr=err.values.T,
    )

    labels = mr.index.to_series()
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0, 2.0)
    ax.axhline(1.0, linestyle=":", color="k", lw=1)

    #     fig.tight_layout()
    fig.set_facecolor("w")
    return fig, ax


mr.sort_values("ML", inplace=True)
plot_standings(mr, title=f"Most Likely Recent $R_t$ by County as of {latest_date}")
plt.savefig("figures/fl_rt.svg", bbox_inches="tight")


# %%
mr.sort_values("High_90", inplace=True)
plot_standings(
    mr, title=f"Most Likely (High) Recent $R_t$ by County as of {latest_date}"
)


# %%
show = mr[mr.High_90.le(1)].sort_values("ML")
fig, ax = plot_standings(show, title=f"Likely Under Control \n {latest_date}")


# %%
show = mr[mr.Low_90.ge(1.0)].sort_values("Low_90")
fig, ax = plot_standings(show, title=f"Likely Not Under Control \n {latest_date}")
# plt.tight_layout(3)
plt.savefig("figures/fl_rt_uncontrolled.svg", bbox_inches="tight")


# %%
rt_url = "data/rt_Florida.csv"
rt_florida = pd.read_csv(rt_url)
latest_rt_florida = rt_florida[rt_florida.date == latest_date]

latest_fl_url = "data/fl_latest.csv"
latest_fl = pd.read_csv(latest_fl_url)

fips_url = "data/florida_county_population.csv"
fips_ref = pd.read_csv(fips_url)

counties_rt = list(latest_rt_florida["county"])
latest_fl = latest_fl[latest_fl.county.isin(counties_rt)]

fips_ref = fips_ref[fips_ref.county.isin(counties_rt)]

fips = fips_ref["FIPS"]
fips = list(fips)


latest_rt_florida["fips"] = fips
latest_rt_florida["county_seat"] = list(fips_ref["county_seat"])
latest_rt_florida["population"] = list(latest_fl["population"])
latest_rt_florida["cases"] = list(latest_fl["cases"])
latest_rt_florida["deaths"] = list(latest_fl["deaths"])
latest_rt_florida["mortality_rate"] = list(latest_fl["mortality_rate"])
latest_rt_florida["cases_rate_per_population"] = list(
    latest_fl["cases_rate_per_population"]
)
latest_rt_florida.reset_index(drop=True)


# %%
latest_rt_florida.to_csv(f"data/latest_rt_florida.csv")


# %%
# url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
url = "https://tinyurl.com/yc4q2rg9"

import json
from urllib.request import urlopen

import plotly.express as px
import plotly.io as pio

with urlopen(url) as response:
    fl = json.load(response)


df = pd.read_csv("data/latest_rt_florida.csv", dtype={"fips": str})


fig = px.choropleth_mapbox(
    df,
    geojson=fl,
    locations="fips",
    color="ML",
    color_continuous_scale=[
        (0, "green"),
        (0.5, "rgb(135, 226, 135)"),
        (0.5, "rgb(226, 136, 136)"),
        (1, "red"),
    ],
    hover_name="county",
    hover_data=[
        "date",
        "county_seat",
        "population",
        "cases",
        "deaths",
        "mortality_rate",
        "cases_rate_per_population",
    ],
    range_color=(0, 2),
    mapbox_style="carto-positron",
    zoom=5.999702795472654,
    center={"lat": 28.03993077755068, "lon": -83.93496716885221},
    opacity=0.8,
    labels={
        "ML": "Most Likely Rₜ value",
        "county_seat": "County Seat",
        "date": "Date",
        "population": "Population",
        "cases": "Cases",
        "deaths": "Deaths",
        "mortality_rate": "Mortality Rate",
        "cases_rate_per_population": "Cases Rate per Population",
        "fips": "FIPS",
    },
    #     width=1024,
    #     height=768,
)

fig.layout.font.family = "Arial"

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(
    width=1000,
    height=1000,
    title=f"{selected_state}: Rₜ per County",
    annotations=[
        dict(
            xanchor="right",
            x=1,
            yanchor="top",
            y=-0.05,
            showarrow=False,
            text="Sources: The New York Times: Coronavirus (Covid-19) Data in the United States, U.S. Census Bureau",
        )
    ],
    autosize=True,
)

fig.show()


# %%
# import plotly.io as pio

# pio.write_json(fig, "map.json")


# %%
# with open("html/florida_rt.html", "w") as f:
#     f.write(fig.to_html(include_plotlyjs="cdn"))


# %%
with open("../danielcs88.github.io/html/florida_rt.html", "w") as f:
    f.write(
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        + fig.to_html(include_plotlyjs="cdn")
    )


# %%
get_ipython().system(" cd ../danielcs88.github.io/ && git pull")


# %%
get_ipython().system(
    ' cd ../danielcs88.github.io/ && git add --all && git commit -m "Update" && git push'
)


# %%
# ! git add --all && git commit -m "Update" && git push
