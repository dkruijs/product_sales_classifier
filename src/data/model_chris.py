#!/usr/bin/env python
# coding: utf-8

# ## Super Eenvoudig Model

# In[44]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix


# In[27]:


data = pd.read_csv('../data/raw/SalesKaggle3.csv')


# In[28]:


data.head()


# We preprocess the data, by encoding MarketingType (the only categorical variable we want to include) as a numerical variable.

# In[77]:


def preproces_data(data, labelencoder_dict):
    data.loc[:,'MarketingTypeD'] = labelencoder_dict['MarketingTypeD'].transform(data.MarketingType)
    
#     data.loc[:,'weights'] = data.agg({'SoldFlag':['mean']}).values[0,0]
#     data.loc[data.SoldFlag == 1,'weights'] = 1 - data.agg({'SoldFlag':['mean']}).values[0,0]
    


# In[71]:


labelencoder = {'MarketingTypeD': preprocessing.LabelEncoder().fit(historical.MarketingType)}


# In[78]:


preproces_data(data, labelencoder)


# In[80]:


historical = data[data.File_Type == 'Historical']
active = data[data.File_Type == 'Active']


# In[81]:


active


# In[82]:


from sklearn.utils.class_weight import compute_class_weight

compute_class_weight('balanced', [0,1], historical.SoldFlag.values)


# We select dependent and independent variables; SoldCount is ommited from the independent variables because the model misunderstands it as a predictor for future sales (where it is NaN). SKU number is omitted, because it is a serial number of the product and thus contains no added information.
# 
# We predict for SoldFlag.

# In[53]:


x_cols = ['ReleaseNumber', 'New_Release_Flag', 'StrengthFactor',
       'PriceReg', 'ReleaseYear', 'ItemCount', 'LowUserPrice', 'LowNetPrice',
       'MarketingTypeD']

y_col = 'SoldFlag'


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(historical[x_cols], historical[y_col].values, test_size=0.30, random_state=42)


# In building the Random Forest classifier we add the argument `class_weight='balanced'` to improve performance; when omitting this, the model performs worse because the data is unbalanced.

# In[97]:


clf = RandomForestClassifier(max_depth=5, random_state=42, criterion='gini', n_estimators=100, verbose=1, class_weight = 'balanced')


# In[101]:


clf.fit(X_train, y_train)


# ## Model Evaluation

# In[50]:


def classifier_model_plot(X_test, y_test, clf, figsize, name):
    
    y_pred=clf.predict_proba(X_test)
    y_pred = y_pred[:,1]
    y_pred_labels=clf.predict(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds2 = precision_recall_curve(y_test, y_pred)

    feats = {} # a dict to hold feature_name: feature_importance

    for feature, importance in zip(X_test.columns, clf.feature_importances_):
        feats[feature] = importance #add the name/value pair 

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances['sd'] = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    
    lw = 2

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=figsize)

    ax1.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.axis(xmin=0,xmax=1.05, ymin=0, ymax=1.05)
    ax1.plot([-0.05, 1.05], [-0.05, 1.05], color='navy', lw=lw, linestyle='--')
    ax1.axis(xmin=0,xmax=1, ymin=0, ymax=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Model ROC')
    ax1.legend(loc="lower right")

    confusion_matrix = plot_confusion_matrix(clf, X_test, y_test, normalize='pred', ax=ax2)
    ax2.set_title('Confusion Matrix')

    ax3.plot(precision, recall, color='darkorange',
             lw=lw, label='Precision Recall Curve')
    ax3.axis(xmin=0,xmax=1.05, ymin=0, ymax=1.05)
    ax3.set_xlabel('Precision')
    ax3.set_ylabel('Recall')
    ax3.set_title('Model PRC')
    ax3.legend(loc="lower right")

    textstr = '\n'.join((
        r'$Accuracy=%.2f$' % (accuracy_score(y_test, y_pred_labels), ),
        r'$Precision=%.2f$' % (precision_score(y_test, y_pred_labels), ),
        r'$Recall=%.2f$' % (recall_score(y_test, y_pred_labels), )))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax3.text(0.10, 0.15, textstr, transform=ax3.transAxes, fontsize=14, bbox=props)



    importances.sort_values(by='Gini-importance').plot.bar(yerr = 'sd', rot=90, ax=ax4)
    ax4.set_title('Importance of Variables')
    
    fig.savefig(name)


# In[96]:


classifier_model_plot(X_test, y_test, clf, (20, 10), 'Modelplaatje.png')


# ## Save trained model for production use

# In[98]:


import pickle


# In[99]:


# Save to file in the current working directory
pkl_filename = "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)


# In[107]:


clf.predict(X_test.iloc[23:24])


# In[108]:


X_test.iloc[23:24]


# In[ ]:




