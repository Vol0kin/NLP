import numpy as np
import pandas as pd

def evaluate_corpus(sequences, sequences_predictions, y_dict, ignore_tag_code=-1):
    import altair as alt
    from sklearn.metrics import f1_score

    """Evaluate classification performance at corpus level, comparing with gold standard.
    Accuracy, F1 score and confusion matrix are printed.
    
    Parameter ignore_tag_code (optional):
        Provides a tag code that does not count for the accuracy (used for instance with the tag 'O')
    """
    # Collect all sequences words labels and predictions in a single list
    fltr_y, fltr_y_hat, y, y_hat = [], [], [], []
    for i in range(len(sequences)):
        tmp_y, tmp_yhat = np.array(sequences[i].y), sequences_predictions[i].y 
        filter_tag = tmp_y != ignore_tag_code
        y.append(tmp_y)
        y_hat.append(tmp_yhat)
        fltr_y.append(tmp_y[filter_tag])
        fltr_y_hat.append(tmp_yhat[filter_tag])
    
    fltr_y, fltr_y_hat, y, y_hat = np.concatenate(fltr_y), np.concatenate(fltr_y_hat), np.concatenate(y), np.concatenate(y_hat)
    
    # Print accuracy and F1 score
    print(f'Accuracy: {np.sum(y == y_hat)/len(y):.3f}')
    print(f"Weighted F1 score: {f1_score(y, y_hat, average='weighted'):.3f}")    
    print(f'Accuracy (ignore tag excluded): {np.sum(fltr_y == fltr_y_hat)/len(fltr_y):.3f}')
    print(f"Weighted F1 score (ignore tag excluded): {f1_score(fltr_y, fltr_y_hat, average='weighted'):.3f}")
    
    # Confusion matrix
    # Create confusion matrix (long format)
    conf_mat = pd.DataFrame(pd.DataFrame({'value': y, 'prediction': y_hat}).value_counts()).reset_index().rename(columns={0: 'num_cases'})
    
    # Map y int values to labels
    reversed_dict = dict(map(reversed, y_dict.items()))
    conf_mat['value'] = [reversed_dict[key] for key in conf_mat['value']]
    conf_mat['prediction'] = [reversed_dict[key] for key in conf_mat['prediction']]
    
    # Scale per number of y-label occurrances
    totals = conf_mat[['value','num_cases']].groupby('value').sum()
    conf_mat['pct_cases'] = conf_mat['num_cases'].values/totals.loc[conf_mat.value,'num_cases'].values
    
    # Plot
    chrt = alt.Chart(conf_mat).mark_circle().encode(
        x=alt.X('prediction:O'),
        y=alt.Y('value:O'),
        color=alt.Color('pct_cases:Q', scale=alt.Scale(scheme='redblue', domain=[0,1])),
        size=alt.Size('pct_cases:Q', scale=alt.Scale(domain=[0,1]), legend=alt.Legend(title='Percentage of words')),
        tooltip=['value','prediction',alt.Tooltip('pct_cases:Q', format='.2%'),'num_cases']
    )
    display(chrt)
    
    return
