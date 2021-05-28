# -*- coding: utf-8 -*-
"""
@author: Ignacio
"""
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,plot_confusion_matrix
#%% ConditionsLabels_dict
"""
Load the dict that contains the name of the five  hydraulic system's
conditions as keys and another dict as values. Values dicts contains the name
of each class as keys and the labels of the classes as values. 
"""
ConditionsLabels_dict = {
    'Cooler condition':{
        'Close to total failure':3,
        'Reduced effifiency':20,
        'Full efficiency':100
    },
    'Valve condition':{
        'Optimal switching behavior':100,
        'Small lag':90,
        'Severe lag':80,
        'Close to total failure':73
    },
    'Pump leakage':{
        'No leakage':0,
        'Weak leakage':1,
        'Severe leakage':2
    },
    'Accumulator condition':{
        'Optimal pressure':130,
        'Slightly reduced pressure':115,
        'Severely reduced pressure':100,
        'Close to total failure':90
    },
    'Stable flag':{
        'Stable' : 0,
        'Not stable' : 1
        #'Conditions were stable':0,
        #'Static conditions might not have been reached yet':1
    }    
}
#%% 
def plt_multiCM(models_dict,X_test,Y_test,condition_name,cmap='Blues'):
    """
     Plotea las matrices de confusion con los resultados de los modelos

    Parameters
    --------------------------------------------------------------------------
    
    models_dict: dict
        Contiene los nombres de los modelos, asociados a los modelos
        entrenados de la forma:
        {'model_name' : model.fit(X_train,Y_train)}
       
    condition_labels: np.array with shape (2205,)
        Array that contains the class label for each instance
        
    condition_name: {'Cooler condition','Valve condition','Pump leakage',
                     'Accumulator condition','Stable flag'}
        Name of hydraulic system's condition to be clasified   
        
    Returns
    --------------------------------------------------------------------------
    out: confusion matrices plots
    
    """    
    classes_names=list(ConditionsLabels_dict[condition_name].keys())
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))   
    for model_name, ax in zip(list(models_dict) , axes.flatten()):
        plot_confusion_matrix(models_dict[model_name], 
                              X_test, 
                              Y_test, 
                              ax=ax, 
                              cmap=cmap,
                              display_labels=classes_names)
        acc = accuracy_score(Y_test,models_dict[model_name].predict(X_test))
        title = model_name + ', accuracy: {:1.3f}'.format(acc)
        ax.title.set_text(title)   
    plt.tight_layout()  
    plt.show()