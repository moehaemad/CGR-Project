import pdb
import pandas as pd
import os
import matplotlib.pyplot as plt
import util

def load_dataset(filename):
    with open(os.path.join (filename), 'rb') as f:
        return pd.read_csv(f, parse_dates = True)
    
def initial_efa(x, num_factors):
    """
    Perform EFA analysis and return factor loadings in the view that is
    appropriate for visual inspection
    """
        #Do EFA using factor analyzer package
    number_of_factors = 12
    fa = util.efa(x, factors=number_of_factors)
    
    #make loadings
        #get rid of factor loading values below a certain threshold
    loadings_reg = fa.loadings
    loadings_to_view = util.trim(fa.loadings, 0.25).abs()
    loadings_corr = loadings_reg.corr().abs()
    
    return fa, loadings_reg, loadings_to_view, loadings_corr

def main():
    read = load_dataset("/media/veracrypt1/State 2.0 CSV.csv")
    filepath = os.path.abspath(__file__)[0:-7]

    other_ind = list(range(0,47))
    other_ind.extend(list(range(73,100)))
    x = read
    #    Do NLP on the responses from the participants
#    nlp.nlp_stats(read, "unique", "Female")
    
    #Report statistics on the participants
    util.report_participant_stats(read, ['Sex', 'Ethnicity', 'Age', 'Yrs_Eng'])
    plt.matshow(x.corr(method="pearson"))
    plt.title("Correlation matrix of State_Con answers")
    plt.savefig(filepath + "/figures/state_con_corr.png")   
    
    #Get rid of columns that aren't questions
    x = util.drop_question(x, other_ind)
    util.bartlett_measure(x)
    number_of_factors = 5
    fa, loadings_reg, loadings_to_view, loadings_corr = initial_efa(x, 
                                                        number_of_factors)
    
    #Calculate mean and plot the graph on the given matrix
    util.mean_graph(x, plot_std="no")
    
    #Do scree test and plot a graph
    anti_image_matrix, kmo_score = util.scree_test(fa, x, number_of_factors)


    #get correlation matrix  to view in variable explorer
    correlation = util.trim(x.corr(), 0.3)
    multicollinearity_corr = util.trim(x.corr(), 0.8)
    
    
    #Do convergent analysis on State Scale 2.0
    """Recall old State Scale instead of State Scale 2.0 """
    state_1 = load_dataset("/media/veracrypt1/State_Mar21_CleanedData.csv")
    indices = list(range(0,47))
    
    indices.extend(list(range(73,117)))
    x_1 = util.drop_question(state_1,indices)
    x_1 = x_1.as_matrix()
    luck_x_1 = x_1
    
    """For convergent analysis """
    state2_ind_luck = [49, 56, 58, 67]
    #do Convergent analysis for luck
    util.convergent_analysis(read.copy(), state2_ind_luck, [32,41],list(
            range(10,22)), x_1, x="Participant", y="State Scores",
    title="Linear Regression on participant scores of Luck") 
    
# =============================================================================
#     Dropping questions
# =============================================================================
    bigls = read
    indices = list(range(0,10))
    indices.extend(list(range(22, 100)))
    bigls = util.drop_question(bigls, indices)
    #indexes are 10-21
    util.drop_question(x, [20]) #remove @21
    #Update: 5 factors still remain
    pdb.set_trace()

if __name__ == "__main__":
    
    main()
    
    
    
    
    
# =============================================================================
# Indexes are... FOR STATE 2.0
# BIGLS questions: 10 -> 21
#     BIGLS total: 22
# 
# GRCS questions: 23 -> 45
#     GRCS total: 46
# 
# State questions: 47 -> 72
#     State total: 73
# 
# Gambling experiences questions: 74 -> 87
#     Gambling experiences total: 88
# 
# Written responses: 89
# =============================================================================
# =============================================================================
# Indexes are... FOR PILOT DATA
# luck questions: 10->21
# luck total: 22
    #BIGLS scale (belief in good luck)
# GRCS(gambling related cognition scale): 23->45
# GRCS total: 46
    #variety: total score -> captures gambling related cognitions
# State_con: 47->72
# State_con total: 73
    #scale being measured (state conviction)
# GE_total: 74->87
# GE_total total: 88
    #gamblign experiences scale (state experiences with gambling)''
# ...and this questions: 89->114
# total: 115
# Written responses: 116
# Indexes that aren't important for correlation matrix 
# [0,1,2,3,4,5,6,7,8,9,22,46,73,88,115]
    
    #Delete b questions
# =============================================================================