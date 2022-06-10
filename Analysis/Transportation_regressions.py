#!/usr/bin/env python
# coding: utf-8

# # The effects of Covid19 on Israel's Public Transportation
# 
# #### Shachar Daya, Ella Tamir, Gal Zivoni 
# #### Advisor: Prof Eran Toch
# ##### Tel - Aviv University, Department of Industrial Engineering

# In the following code we used three regression models - Linear, Ridge & Lasso.
# The data that was used describes the number of travels in buses in israel during the years 2019 - 2021. This information is public and can be found at https://data.gov.il/dataset/tikufim-2021.
# 
# Information about the Data concerning Covid19, Unemployment, Stringency index can be found at https://github.com/GalZivoni/TAU-2022-Project-Public-Transportation.
# 
# Further information, Analysis and discussion can be found (in hebrew) in the document which is in the git repository above.

# In[1]:


# Imports:
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, RidgeCV, LarsCV, Ridge, Lasso, LassoCV
from regressors import stats  
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Reading the data
df = pd.read_excel('Data_significance_national.xlsx')
df


# In[3]:


value = df[["Value"]]
labels = ["Avg weekly percentage Covid19 positives", "Avg weekly new Covid19 positives", "Avg weekly critical patients",
          "Total hospitalized", "Avg Stringency Index", "Avg daily stringency lockdown", 
          "Avg daily percentage change work-place arrival", "Avg daily stringency schools", 
          "Avg daily stringency work-place restrictions", "No. of workdays", 
          "Avg monthly unemployment percentage adjusted", "Early", "Intact", "Late", "Not Authorized",
          "Not Performed", "Weekly bus travels"]

features = df[labels]


# In[4]:


features


# In[5]:


value


# In[6]:


# Converting to numpy arrays:
features_np = features.to_numpy()
value_np = value.to_numpy()


# ### Linear Regression: 

# Using the Sklearn model for linear regression.

# In[7]:


Linear_reg = linear_model.LinearRegression()
Linear_reg.fit(features_np, value_np)
linear_r_sq = Linear_reg.score(features_np, value_np)

# Printing R^2
print('coefficient of determination:', linear_r_sq)


# In[8]:


linear_adj_r_sq = stats.adj_r2_score(Linear_reg, features_np, value_np)

# Printing R^2 adjusted
print('adjusted coefficient of determination:', linear_adj_r_sq)


# In[9]:


linear_reg_intercept = Linear_reg.intercept_
print("Linear regression intercept is: ", linear_reg_intercept)


# In[10]:


linear_reg_coef = Linear_reg.coef_
print("Linear regression coefs are: ", linear_reg_coef)


# In[11]:


new_labels = ['inter']
new_labels += [label for label in labels]


# In[12]:


# Matching the coefs with the labels
coefs = linear_reg_coef[0].tolist()
d = {'Var': labels, 'coef': coefs}
coefs_pd = pd.DataFrame(d)
coefs_pd


# In[13]:


# P values
p_values_regression = stats.coef_pval(Linear_reg, features_np, value_np)
regression_p_values = pd.DataFrame(list(zip(new_labels, p_values_regression)), columns =['Features','P_value'])
regression_p_values


# In[14]:


# Significant variables
features_regression = regression_p_values[regression_p_values.P_value <0.05]
features_regression


# ### Ridge Regression:

# Using the sklearn model.

# In[15]:


Ridge_reg = Ridge()
Ridge_reg.fit(features_np, value_np)
Ridge_reg.get_params()
ridge_r_sq = Ridge_reg.score(features_np, value_np)
print('regression score:', ridge_r_sq)


# In[16]:


ridge_adj_r_sq = stats.adj_r2_score(Ridge_reg, features_np, value_np)
print('adjusted coefficient of determination:', ridge_adj_r_sq)


# In[17]:


Ridge_reg_intercept = Ridge_reg.intercept_
print("Ridge regression intercept is: ", Ridge_reg_intercept)


# In[18]:


Ridge_reg_coef = Ridge_reg.coef_
print("Ridge regression coefs are: ", Ridge_reg_coef)


# In[19]:


coefs = Ridge_reg_coef[0].tolist()
d = {'Var': labels, 'coef': coefs}
coefs_pd = pd.DataFrame(d)
coefs_pd


# In[20]:


new_labels = ['inter']
new_labels += [label for label in labels]
p_values_ridge = stats.coef_pval(Ridge_reg, features_np, value_np)

Ridge_p_values = pd.DataFrame(list(zip(new_labels, p_values_ridge)), columns =['Features','P_value'])
Ridge_p_values


# In[21]:


features_ridge = Ridge_p_values[Ridge_p_values.P_value <0.05]
features_ridge


# ### Lasso Regression: 

# In[22]:


Lasso_reg = Lasso()
Lasso_reg.fit(features_np, value_np)
Lasso_reg.get_params()
lasso_r_sq = Lasso_reg.score(features_np, value_np)
print('regression score:', lasso_r_sq)


# In[23]:


lasso_adj_r_sq = stats.adj_r2_score(Lasso_reg, features_np, value_np)
print('adjusted coefficient of determination:', lasso_adj_r_sq)


# In[24]:


Lasso_reg_intercept = Lasso_reg.intercept_
Lasso_reg_intercept


# In[25]:


Lasso_reg_coef = Lasso_reg.coef_
Lasso_reg_coef


# In[26]:


coefs = Lasso_reg_coef.tolist()
d = {'Var': labels, 'coef': coefs}
coefs_pd = pd.DataFrame(d)
coefs_pd


# In[27]:


new_labels = ['inter']
new_labels += [label for label in labels]
p_values_lasso = stats.coef_pval(Lasso_reg, features_np, value_np)

Lasso_p_values = pd.DataFrame(list(zip(new_labels, p_values_lasso)), columns =['Features','P_value'])
Lasso_p_values


# In[28]:


features_lasso = Lasso_p_values[Lasso_p_values.P_value <0.05]
features_lasso


# ## Comparison: 

# ### 1) Comparing R results: 

# In[29]:


regressions = ["Linear", "Ridge", "Lasso"]
# r_sq = [linear_r_sq, ridge_r_sq, lasso_r_sq]
# adj_r_sq = [linear_adj_r_sq, ridge_adj_r_sq, lasso_adj_r_sq]
r_sq = [round(i*100,2) for i in (linear_r_sq, ridge_r_sq, lasso_r_sq)]
adj_r_sq = [round(i*100, 2) for i in (linear_adj_r_sq, ridge_adj_r_sq, lasso_adj_r_sq)]
adj_r_sq


# In[30]:


reg_r_sq_res = pd.DataFrame(list(zip(r_sq, adj_r_sq)), columns =['r_sq', 'adj_r_sq'])
reg_r_sq_res.index = ['Linear', 'Ridge', 'Lasso']
reg_r_sq_res


# In[31]:


plt.style.use('ggplot')
fig, ax = plt.subplots(figsize = (15,5))
x = np.arange(len(regressions))  # the label locations
width = 0.35  # the width of the bars

rects1 = ax.bar(x - width/2, r_sq, width, label='r_sq')
rects2 = ax.bar(x + width/2, adj_r_sq, width, label='adj_r_sq')
ax.set_ylabel('Scores', size = 20)
ax.set_title('Scores by regression', size = 30)
ax.set_xlabel('Regressions', size = 20)
ax.set_xticks(x)
ax.set_xticklabels(labels =["Linear", "Ridge", "Lasso"], size = 15)
ax.legend(fontsize=20)


ax.bar_label(rects1, padding=5, label_type='center', size = 18)
ax.bar_label(rects2, padding=3, label_type='center', size = 18)

fig.tight_layout()

plt.show()


# ### 2) Comparing P values:

# In[32]:


p_values = pd.DataFrame(list(zip(p_values_regression, p_values_ridge, p_values_lasso)), columns = ["regression_p_values", "Ridge_p_values", "Lasso_p_values"])
p_values.index = new_labels
p_values


# In[33]:


def distinct_feature(cell_value):

    highlight = 'background-color: yellow;'
    default = ''

    if type(cell_value) in [float, int]:
        if cell_value <0.05:
            return highlight
    return default

p_values.style.applymap(distinct_feature)

