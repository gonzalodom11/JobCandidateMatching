from data_representation import *
from fairlearn.metrics import MetricFrame, selection_rate
import matplotlib.pyplot as plt

def fairness_assesment(interactions, interactions_not5, n = 10000):    
    sex_df = stats_graph(interactions, interactions_not5, n)
    
    y_true = sex_df["Match"]
    sex = sex_df["Sex"]

    matches = sex_df[sex_df["Match"]==1]
    
    print("General ratio:",matches.shape[0]/sex_df.shape[0])


    selection_rates = MetricFrame(
        metrics=selection_rate, y_true=y_true, y_pred=y_true, sensitive_features=sex
    )

    fig = selection_rates.by_group

    fig.plot.bar(
        legend=False, rot=0, title="Fraction of interactions to get an interview/match"
    )

    plt.show()



fairness_assesment(interactions, interactions_not5,)