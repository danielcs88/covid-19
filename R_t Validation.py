#!/usr/bin/env python
# coding: utf-8

# # $R_t$ Validation: Florida

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic("config", "IPCompleter.greedy=True")
get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
from IPython.display import clear_output


# In[2]:


rt_url = "data/rt_Florida.csv"
rt_florida = pd.read_csv(rt_url)
rt_florida


# In[3]:


miami_rt = rt_florida[(rt_florida.county == "Miami-Dade")]
miami_rt
miami_rt.to_csv("data/miami_rt.csv", index=False)


# In[4]:


url_counties = (
    "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
)

timeline = pd.read_csv(url_counties)
florida = timeline[(timeline.fips >= 12000) & (timeline.fips < 13000)]
florida.to_csv("data/florida.csv", index=False)


# In[5]:


miami = timeline[(timeline.fips == 12086)]
miami.to_csv("data/miami_cases.csv", index=False)


# In[6]:


fl_url = "data/florida.csv"
fl_cases = pd.read_csv(fl_url)
fl_cases


# In[7]:


miami.info()


# In[8]:


miami["k"] = miami["cases"].diff()
miami.head()


# In[9]:


miami_rt.head()


# In[10]:


miami = miami.set_index(miami["date"])
miami = miami[["county", "cases", "k", "deaths"]]
miami.tail()


# In[11]:


# miami.drop(miami.head(4).index, inplace=True)


# In[12]:


miami = miami.merge(miami_rt, left_on="date", right_on="date")
miami = miami.drop(["county_y"], axis=1)


# In[13]:


miami = miami.rename(columns={"county_x": "county"})


# In[14]:


miami.tail()


# In[15]:


miami


# In[16]:


# Visualizing correlation with Seaborn
# sns.set(rc={"figure.figsize": (30, 21)})
# sns.set(font_scale=1.5)  # crazy big
sns.heatmap(miami.corr(), cmap="seismic", annot=True, vmin=-1, vmax=1)


# In[17]:


# sns.set(rc={"figure.figsize": (40, 28)})
# sns.set(font_scale=1)  # crazy big
g = sns.pairplot(miami, vars=["ML", "k", "cases"], palette="husl")


# In[18]:


sns.set(font_scale=4)  # crazy big
sns.jointplot(x="ML", y="k", data=miami, kind="kde", height=28, ratio=2)


# I coded the $\text{rt_validity}$ like this because I want the probability that the outcome is 0

# In[19]:


k_list = list(miami["k"])

ml_list = list(miami["ML"])

k_decrease = []

for elem in k_list:
    if elem < k_list[k_list.index(elem) - 1]:
        k_decrease.append(1)
    else:
        k_decrease.append(0)

ml_decrease = []

for elem in ml_list:
    if elem < ml_list[ml_list.index(elem) - 1]:
        ml_decrease.append(1)
    else:
        ml_decrease.append(0)


rt_validity = []

for x, y in zip(k_decrease, ml_decrease):
    if x == y:
        rt_validity.append(1)
    else:
        rt_validity.append(0)


miami["rt_validity"] = rt_validity
miami["ml_decrease"] = ml_decrease
miami["k_decrease"] = k_decrease


# In[20]:


miami.tail(39)


# In[21]:


miami.set_index("date")


# In[22]:


# import datetime

fixed_dates_df = miami.copy()
fixed_dates_df["date"] = fixed_dates_df["date"].apply(pd.to_datetime)
fixed_dates_df = fixed_dates_df.set_index(fixed_dates_df["date"])
fixed_dates_df = fixed_dates_df[["cases"]]
fixed_dates_df


# In[23]:


import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

sns.set(font_scale=1)

register_matplotlib_converters()
plt.style.use("ggplot")
fixed_dates_df.plot(color="purple")


# In[24]:


from statsmodels.tsa.seasonal import seasonal_decompose

sns.set(font_scale=0.7)
result = seasonal_decompose(fixed_dates_df)
fig = result.plot()


# In[25]:


fixed_dates_df.info()


# In[26]:


from fbprophet import Prophet

model = Prophet()
train_df = fixed_dates_df.rename(columns={"cases": "y"})
train_df["ds"] = train_df.index
model.fit(train_df)


# In[27]:


pd.plotting.register_matplotlib_converters()
future = model.make_future_dataframe(12, freq="M", include_history=True)
forecast = model.predict(future)
model.plot(forecast)


# ## OLS
#
# $$k:= \text{new cases of COVID-19}$$
#
# $$k_{\text{decrease}}:= 1 ~ \text{if} ~ k_t < k_{t-1}$$
# $$\quad \text{else} ~ 0$$
#
# $$ML:= \text{most likely} R_t \text{value}$$
# $$ml_{\text{decrease}}:= 1 ~ \text{if} ~ ML_t < ML_{t-1}$$
# $$\quad \text{else} ~  0$$
#
# $$Low_{90} = \text{Low 90% probable range of ML}$$
# $$High_{90} = \text{High 90% probable range of ML}$$
# $$Low_{50} = \text{Low 50% probable range of ML}$$
# $$High_{50} = \text{High 50% probable range of ML}$$
#
# $$\theta = \text{unobserved variables}$$
#
# $$k = \theta + ML + Low_{90} + High_{90} + Low_{50} + High_{50}$$

# In[28]:


model = smf.ols("k ~ ML + Low_90 + High_90 + Low_50 + High_50", data=miami).fit()
print(model.summary2())


# In[29]:


miami["ols_pred"] = model.predict()


# In[30]:


miami.tail(10)


# ## Logit Model
#
# $$P(k_{\text{decrease}} ~|~ k + ml_{decrease})$$

# In[31]:


logit = smf.logit("k_decrease ~ k + C(ml_decrease)", data=miami).fit()


# In[32]:


print(logit.summary2())


# In[33]:


print(logit.get_margeff(at="mean", method="dydx").summary())


# In[34]:


miami["logit_prob"] = logit.predict()
miami.tail()


# In[35]:


miami["logit_pred"] = [0 if x < 0.5 else 1 for x in miami["logit_prob"]]
miami.tail()


# In[36]:


cm = pd.crosstab(miami["rt_validity"], miami["logit_pred"], margins=True)
cm


# In[37]:


TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]
TP = cm[1][1]

accuracy = (TP + TN) / len(miami)
error = 1 - accuracy
sensitivity = TP / (FN + TP)
specificity = TN / (TN + FP)

print("Classification Statistics")
print("=========================")
print("Accuracy:\t", round(accuracy, 4))
print("Error:\t\t", round(error, 4))
print("Sensitivity:\t", round(sensitivity, 4))
print("Specificity:\t", round(specificity, 4))
