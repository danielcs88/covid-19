<meta name="viewport" content="width=device-width, initial-scale=1.0">

# COVID-19: Florida

I have created two Jupyter notebooks.

1. [Descriptive statistics of the ongoing Covid-19 pandemic](COVID-19.ipynb)

   I have included statistics from the metropolitan area of Miami (Miami-Dade,
   Broward, Palm Beach) as well as top counties in the United States, top
   countries around the world, Colombia, and Venezuela.

   Additionally, I have created graphs to illustrate the exponential growth of
   both the cases of COVID-19 as well as Initial Unemployment Cases.

2. [Florida R₀ Analysis per County](https://github.com/danielcs88/covid-19/blob/master/Florida%20R_0.py)

   An analysis of Florida per county level to make predictions on the
   possibility of new cases, the implementation is using
   [Kevin Systrom's model](https://github.com/k-sys/covid-19),
   and using the adaptation by
   [Ashutosh Sanzgiri per county](https://github.com/k-sys/covid-19/blob/e95ae71f1eea827baffce2d308f767634951f9e3/Realtime_R0_by_county.ipynb)
   per county to analyze the situation for Florida and its counties.

## Requirements

- [Plotly](https://plotly.com/python/)

## Data Sources

- [New York Times: Coronavirus (Covid-19) Data in the United States](https://github.com/nytimes/covid-19-data)
- [Johns Hopkins CSSE: 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data](https://github.com/CSSEGISandData/COVID-19/)
- [Initial Claims (ICSA): FRED, St. Louis Fed](https://fred.stlouisfed.org/series/ICSA)

## Implementation Sources

- [Kevin Systrom: covid-19](https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb)
- [Ashutosh Sanzgiri: covid-19](https://github.com/sanzgiri/covid-19/blob/master/Realtime_R0_by_county.ipynb)

## [Dashboard](https://danielcs88.github.io/covid-19.html)

I have created a dashboard hosted on my website with key figures:

- Total cases in the United States
- Total cases in Florida (and Miami-Dade)
- Miami-Dade | New Cases per Day
- Miami-Dade | Real-time Rₜ
- Florida | Rₜ per County | Timeline
- Florida | Rₜ per County | Latest Values
- Florida | Rₜ per County Map
