#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


hitters = pd.read_csv("Assignment_3_Hitters.csv")


# ## Data Preprocessing & Cleaning
# We want to fill in missing salary data instead of just dropping the missing data rows. One way to do this is by filling in the mean salary (imputation).
# However we can be smarter about this and check the average sal by NewLeague. For example:
# 

# Avg salaries in the NewLeague A is slightly higher than Avg salaries in the NewLeague N

# In[3]:


print(round(hitters[["Salary","NewLeague"]][hitters["NewLeague"] =="A"].mean()))


# In[4]:


print(round(hitters[["Salary","NewLeague"]][hitters["NewLeague"] =="N"].mean()))


# In[5]:


# Heatmap to check for the missing values
plt.figure(figsize=(12, 7))
sns.heatmap(hitters.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


def impute_sal(cols):
    
    sal = cols[0]
    league = cols[1]
        
    if pd.isnull(sal):

        if league == "A":
            return 537

        else:
            return 535

    else:
        return sal


# In[7]:


hitters['Salary'] = hitters[['Salary','NewLeague']].apply(impute_sal,axis=1)


# Now let's check that heat map again!

# In[8]:


plt.figure(figsize=(12, 7))
sns.heatmap(hitters.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Converting Categorical Features 
# 
# We'll need to convert categorical features to dummy variables using pandas! Otherwise our machine learning algorithm won't be able to directly take in those features as inputs.

# In[9]:


hitters.info()


# In[10]:


league = pd.get_dummies(hitters['League'],drop_first=True)
division = pd.get_dummies(hitters['Division'],drop_first=True)


# In[11]:


hitters.drop(['League','Division','Unnamed: 0'],axis=1,inplace=True)


# In[12]:


hitters = pd.concat([hitters,league,division],axis=1)


# In[13]:


X = hitters.drop(["Salary","NewLeague"],axis=1)
y = hitters['Salary']


# In[14]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[15]:


X = np.array(X)
y = np.array(y)


# ## Train Test Split

# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[18]:


# Initialize the weights and bias randomly
def initialize_parameters(X):
    b = np.random.rand()
    w = np.random.rand(X.shape[1],1)
    return w, b


# In[19]:


def model(X, y, lr, iterations, lamb, regular):
    
    costs = []

    counter = 0
    n = len(X)
    w, b = initialize_parameters(X)
    
   
    while iterations > counter:
        
        
        z = np.dot(w.T, X.T) + b
        
        if z.all() >= 0:
            pred = z
            
        else:
            pred = z * 0.05
            
        #Calculate Loss Function
        MSE = np.square(np.subtract(y, pred)).mean()
        
        if regular == 'L1':
            
            cost = MSE + (lamb * abs(np.sum((w))))
            dw = 1/n * (np.dot(X.T, (pred - y).T) + lamb)
            db = 1/n* np.sum(pred - y)

            w = w - lr * dw
            b = b - lr * db
            
        elif regular == 'L2':
            cost = MSE + ((lamb / 2) * (np.sum((w)**2)))
            
            dw = 1/n * (np.dot(X.T, (pred - y).T) + lamb)
            db = 1/n* np.sum(pred - y)

            w = w - lr * dw
            b = b - lr * db
            
        else:
            cost = MSE
            dw = 1/n * np.dot(X.T, (pred - y).T)
            db = 1/n* np.sum(pred - y)

            w = w - lr * dw
            b = b - lr * db
            
        if counter % 10 == 0:
            costs.append(cost)
            
                
        counter+= 1
            
    return w, b, costs


# In[20]:


def plot(cost, dataset):
    plt.figure(figsize = (15,7))
    sns.lineplot(x = list(range(0,len(cost))), y = cost)
    plt.title("MSE - "+ dataset)
    plt.xlabel("# of iterations")
    plt.ylabel("MSE")
    plt.show()


# In[ ]:





# In[21]:


cost_lr1 = []
w, b, cost_lr1 = model(X_train, y_train, lr = 0.001, iterations = 1000, lamb = 0.01, regular = "L1")

cost_lr2 = []
w, b, cost_lr2 = model(X_train, y_train, lr = 0.01, iterations = 1000, lamb = 0.01, regular = "L1")

cost_lr3 = []
w, b, cost_lr3 = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 0.01, regular = "L1")

fig = plt.figure(figsize = (10,7))

ax = fig.add_axes([0,0,1,1])

ax.plot(list(range(0,len(cost_lr1))), cost_lr1, label="0.001")
ax.plot(list(range(0,len(cost_lr2))), cost_lr2, label="0.01")
ax.plot(list(range(0,len(cost_lr3))), cost_lr3, label="0.1")
ax.legend()
ax.set_title("L1 Regulization and lamda = 0.01")


# In[22]:



cost1 = []
w, b, cost1 = model(X_train, y_train, lr = 0.001, iterations = 1000, lamb = 10, regular = "L1")

cost2 = []
w, b, cost2 = model(X_train, y_train, lr = 0.01, iterations = 1000, lamb = 10, regular = "L1")

cost3 = []
w, b, cost3 = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 10, regular = "L1")


fig = plt.figure(figsize = (10,7))

ax = fig.add_axes([0,0,1,1])

ax.plot(list(range(0,len(cost1))), cost1, label="0.001")
ax.plot(list(range(0,len(cost2))), cost2, label="0.01")
ax.plot(list(range(0,len(cost3))), cost3, label="0.1")
ax.legend()
ax.set_title("L1 Regulization and lamda = 10")


# In[ ]:





# # Question b

# In[23]:


w, b, cost_l1 = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 0.01, regular = "L1")
w, b, cost_l2 = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 0.01, regular = "L2")

fig = plt.figure(figsize = (10,7))

ax = fig.add_axes([0,0,1,1])

ax.plot(list(range(0,len(cost_l1))), cost_l1, label="L1")
ax.plot(list(range(0,len(cost_l2))), cost_l2, label="L2")

ax.legend()
ax.set_title("MSE + Regulization and lamda = 0.01")


# In[24]:


w, b, cost_l10 = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 10, regular = "L1")
w, b, cost_l20 = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 10, regular = "L2")

fig = plt.figure(figsize = (10,7))

ax = fig.add_axes([0,0,1,1])

ax.plot(list(range(0,len(cost_l10))), cost_l10, label="L1")
ax.plot(list(range(0,len(cost_l20))), cost_l20, label="L2")

ax.legend()
ax.set_title("MSE + Regulization and lamda = 10")


# # Question c

# In[30]:


w, b, cost = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 0.01, regular = " ")
print(w)
print(b)


# In[31]:


w1, b, cost = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 0.01, regular = "L2")
print(w1)
print(b)


# In[32]:


w1, b, cost = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 10, regular = "L2")
print(w1)
print(b)


# In[33]:


w1, b, cost = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 0.01, regular = "L1")
print(w1)
print(b)


# In[34]:


w2, b, cost = model(X_train, y_train, lr = 0.1, iterations = 1000, lamb = 10, regular = "L1")
print(w1)
print(b)


# In[ ]:





# In[ ]:




