# -*- coding: utf-8 -*-
import os
import pickle
import click
import logging
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix

def classifier_model_plot(X_test, y_test, clf, figsize, output_filepath, name):
    """
    Create several plots to evaluate model performance
    """
    
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
    
    fig.savefig(os.path.join(output_filepath, name))


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    This module trains the Random Forest Classifier model.
    """

    historical = pd.read_csv(input_filepath)

    x_cols = ['ReleaseNumber', 'New_Release_Flag', 'StrengthFactor',
        'PriceReg', 'ReleaseYear', 'ItemCount', 'LowUserPrice', 'LowNetPrice',
        'MarketingTypeD']

    y_col = 'SoldFlag'

    X_train, X_test, y_train, y_test = train_test_split(historical[x_cols], 
                                                        historical[y_col].values, 
                                                        test_size=0.30, 
                                                        random_state=42)

    clf = RandomForestClassifier(max_depth=5, 
                                random_state=42, 
                                criterion='gini', 
                                n_estimators=100, 
                                verbose=1, 
                                class_weight = 'balanced')

    clf.fit(X_train, y_train)

    # Evaluate model
    classifier_model_plot(X_test, y_test, clf, (20, 10), output_filepath, 'Model_evaluation_plots.png')

    # Save to file
    pkl_filename = "model.pkl"
    with open(os.path.join(output_filepath, pkl_filename), 'wb') as file:
        pickle.dump(clf, file)