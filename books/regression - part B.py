#!/usr/bin/env python
# coding: utf-8

# In[170]:


from sklearn import model_selection, linear_model, metrics, naive_bayes
import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[171]:


get_ipython().run_line_magic('store', '-r X_train_perfect')
get_ipython().run_line_magic('store', '-r X_test_perfect')
get_ipython().run_line_magic('store', '-r y_train_perfect')
get_ipython().run_line_magic('store', '-r y_test_perfect')
get_ipython().run_line_magic('store', '-r X')
get_ipython().run_line_magic('store', '-r y')

print(X_test_perfect[0])


# In[172]:


K=10
outerCV = model_selection.KFold(n_splits=K)
innerCV = model_selection.KFold(n_splits=K)


# In[173]:


## Outer folds are to compare performance between models (use X_train_outer)
## Inner folds are to achieve the best score in each of the models (by testing different parameters)


# In[179]:


# regularized linear regression
n_alphas = 10
alphas = np.logspace(-10, 10, n_alphas)
ridge_inner_error = []
min_ridge_error = 1

# neural network
nn_error = []



for outer_train_index, outer_test_index in outerCV.split(X):
    outer_number = 1
    
    # Outer folds
    X_train_outer, y_train_outer = X[outer_train_index,:], y[outer_train_index]
    X_test_outer, y_test_outer = X[outer_test_index,:], y[outer_test_index]
    
    # Inner folds
    for inner_train_index, inner_test_index in outerCV.split(X_train_outer):
        X_train_inner, y_train_inner = X[inner_train_index,:], y[inner_train_index]
        X_test_inner, y_test_inner = X[inner_test_index,:], y[inner_test_index]
        
        # linear regression (no parameters to test)
        linreg = linear_model.LinearRegression().fit(X_train_inner, y_train_inner)
        y_pred_linreg = linreg.predict(X_test_inner)
#         print('linear regression -> error:', metrics.mean_squared_error(y_test_inner, y_pred_linreg))
        
        # regularized linear regression (alternate different lambdas)
        for a in alphas:
            ridge = linear_model.Ridge(alpha=a, fit_intercept=True)
            ridge.fit(X_train_inner, y_train_inner)
            y_pred_ridge = ridge.predict(X_test_inner)
            ridge_inner_error.append({
                'error': metrics.mean_squared_error(y_test_inner, y_pred_ridge),
                'alpha': a
            })
            
            
        for item in ridge_inner_error:
            # print(item)
            if (item['error'] < min_ridge_error):
                min_ridge_error = item['error']
            if (item['error'] == min_ridge_error):
                min_ridge_alpha = item['alpha']
    

        print('ridge ->', 'error:', min_ridge_error,', lambda', min_ridge_alpha)
    


# In[ ]:




