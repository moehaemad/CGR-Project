import os
import pandas as pd
import matplotlib.pyplot as plt
import util

def load_dataset(filename):
    with open(os.path.join(filename), 'rb') as f:
        return pd.read_csv(f, parse_dates=True)
def prep_state1(x, mask):
    """
    input:
        x (DataFrame) is the survey we want to extract questions from
        mask (list) is the indices for which the arrays are taken from
    """
    x = x.copy()
    #return so that I can index as arrays are indexed instead of question #'s
    mask = [x - 1 for x in mask]
    toreturn = x.iloc[:, mask].as_matrix()
    return toreturn
def initial_efa(x, num_factors):
    """
    Perform EFA analysis and return factor loadings in the view that is
    appropriate for visual inspection
    """
    #Do EFA using factor analyzer package
    number_of_factors = num_factors
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
    elim_ind = list(range(0, 47))
    elim_ind.extend(list(range(73, 100)))
    x = read
    #    Do NLP on the responses from the participants
    #    nlp.nlp_stats(read, "unique", "Female")
    #Report statistics on the participants
    util.report_participant_stats(read, ['Sex', 'Ethnicity', 'Age', 'Yrs_Eng'])
    plt.matshow(x.corr(method="pearson"))
    plt.title("Correlation matrix of State_Con answers")
    plt.savefig(filepath + "/figures/state_con_corr.png")
    #Get rid of columns that aren't questions
    x = util.drop_question(x, elim_ind)
    util.bartlett_measure(x)
    number_of_factors = 5
    fa, loadings_reg, loadings_to_view, loadings_corr = initial_efa(x, number_of_factors)
    plt.close()
    plt.matshow(loadings_to_view)
    plt.title("Visual representation of the factor loadings")
    plt.xlabel("Factors")
    plt.ylabel("Questions")
    plt.savefig(filepath + "/figures/factor_loadings.png")
    #Calculate mean and plot the graph on the given matrix
    util.mean_graph(x, plot_std="no")
    #Do scree test and plot a graph
    anti_image_matrix, kmo_score = util.scree_test(fa, x, number_of_factors)
    #get correlation matrix  to view in variable explorer
    correlation = util.trim(x.corr(), 0.3)
    multicollinearity_corr = util.trim(x.corr(), 0.8)
    #Get Cronbach alpha coefficients
    state2_cr = util.cronbach_alpha(read.iloc[:, 47:72])
    bigls_cr = util.cronbach_alpha(read.iloc[:, 10:21])
    grcs_cr = util.cronbach_alpha(read.iloc[:, 23:45])
    print(state2_cr)
    print(bigls_cr)
    print(grcs_cr)
    #Do convergent analysis on the old state scale (1.0)
    state_1 = load_dataset("/media/veracrypt1/State_Mar21_CleanedData.csv")
    indices = list(range(0, 47))
    indices.extend(list(range(73, 117)))
    x_1 = util.drop_question(state_1, indices)
    state1_cr = util.cronbach_alpha(state_1.iloc[:, 47:72])
    print(state1_cr)
    #indices for luck in old state scale
    state1_indices = [[1, 13, 26], [4, 10, 12, 20], [2, 3, 7, 11, 14, 22, 23],
                      [5, 8, 15], [6, 9, 17, 21, 25]]
    luck_x_1 = prep_state1(x_1, state1_indices[1])
    ioc_x_1 = prep_state1(x_1, state1_indices[2])
    ge_x_1 = prep_state1(x_1, state1_indices[0]+state1_indices[3])
    #0 -> GRCS-IB: interpretive control/bias
    #1 -> GRCS-IC: illusion of control
    #2 -> GRCS-PC: predictive control
    #3 -> GRCS-GE: gambling related expectancies
    #4 -> GRCS-IS: perceived inability to stop gambling
    grcs = ['GRCS-IB: interpretive control/bias', 'GRCS-IC: illusion of control',
            'GRCS-PC: predictive control', 'GRCS-GE: gambling related expectancies',
            'GRCS-IS: perceived inability to stop gambling']
    grcs_indices = [[-1], [-2], [-3], [-4], [-5]]
    #0 -> Hot hand fallacy
    #1 -> Belief in luck
    #2 -> Illusion of Control
    #3 -> Gamblers fallacy
    #4 -> Anthropomorphic
    #5 -> Supernatural
    state2_names = ['Hot Hand Fallacy', 'Belief in luck', 'Illusion of Control',
                    'Gambler\'s fallacy', 'Athropomorphic', 'Supernatural']
    state2_indices = [[47, 59, 72], [49, 56, 58, 67], [48, 52, 54, 57, 61, 69, 70],
                      [52, 52, 55, 64], [51, 53, 63, 66, 68], [60, 62, 65, 71]]
    #Old state indices have the same names as state 2.0 names
#    state1_indices = [[47,59,72], [56,58,66], [48,49,53,57,60,68,69], [51,54,61],
#                      [52,55,63,67,71], [50,62,64,65,70]]
    """For convergent analysis """
#    state2_ind_luck = [49, 56, 58, 67]
    #do Convergent analysis for luck
    util.convergent_analysis(read.copy(), state2_indices[1], [32, 41], list(range(10, 22)),
                             luck_x_1, x="Participant", y="State Scores",
                             title="Linear Regression on participant scores of Luck",
                             luck="yes", savefig="luck")
    #do Convergent analysis for illusion of control
    plt.close()
    util.convergent_analysis(read.copy(), state2_indices[2], grcs_indices[1],
                             [], ioc_x_1, x="Participant", y="State Scores",
                             title="Linear Regression on participant scores of"+
                             "IOC", luck="no", savefig=state2_names[2])
    plt.close()
    util.convergent_analysis(read.copy(), state2_indices[0]+state2_indices[3],
                             grcs_indices[3], [], ge_x_1, x="Participant",
                             y="State Scores", title="Linear Regression on" +
                             "participant scores of " +
                             "Gambling Related Expectancies", luck="no",
                             savefig="Gambling" + " related expectancies")
# =============================================================================
#     Dropping questions: part of efa procedure
# =============================================================================
    bigls = read
    indices = list(range(0, 10))
    indices.extend(list(range(22, 100)))
    bigls = util.drop_question(bigls, indices)
    #indexes are 10-21
    util.drop_question(x, [20]) #remove @21
    #Update: 5 factors still remain
if __name__ == "__main__":
    main()
    