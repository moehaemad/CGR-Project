import numpy as np
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import scipy.stats as ss
import pdb
import famissing
import os

my_path = os.path.abspath(__file__)[0:-7]

def score(x):
    """
    purpose is to collapse all the item scores in each questionaire into one
    input: df (DataFrame) 
    output: toreturn (list) list of scores for each scale
    """
    toreturn = []
    for i in range(x.shape[0]):
        toreturn.append(np.sum(x[i,:]))
    return toreturn

def put_together(x, y):
    """
    Make list of tuples such that the x values are unique and y values are an
        average. This is to graph state as x-axis and other scales as y_axis
        to show linear relationship
    input:
        x (list) of state score answers
        y (list) of grcs/bigls/old state score answers
    output:
        to_return (list) of x/y combined as per mentioned above
    """
    toreturn = []
    tuples = list(zip(x,y))
    #can sort on other axis via sorted(list, key=lambda x:x[1])
    tuples = sorted(tuples)
    last = 0
    temp = []
    for i in tuples:
        if (i[0] == last):
            temp.append(i[1])
        elif (i[0] != last):
            toreturn.append((last,np.average(temp)))
            last = i[0]
    pdb.set_trace()
    return toreturn

def plot_conv(x_plot, y_plot, *args, **kwargs):
    """
    Plot all of the variables in the two lists in the form of x matrix and y
        matrix
    
    input: 
        x_plot (list) of variables of participant responses
        y_plot (list) of scores
    output: none
    """
    
    
    #Make input into list x_y_list
    colours = ['blue','red', 'purple', 'black']
    line_names = args[0]
    #######################TEMPORARY###########################################
    state = y_plot[0]
    bigls = y_plot[1]
    newscore = put_together(state, bigls)
    plt.close()
    for i in range(len(x_plot)):
        y = y_plot[i]
        x_range = np.arange(len(y))
        m, b = np.polyfit(x_range, y, deg=1)
        fx = b + m*x_range
        plt.plot(x_range, fx, color=colours[i], label=line_names[i])
#    pdb.set_trace()
    plt.legend(loc='upper right')
    plt.xlabel(kwargs.get("x_name"))
    plt.ylabel(kwargs.get("y_name"))
    plt.title(kwargs.get("title"))
    plt.show()
    
    return 0

def convergent_analysis(df, ind_state, ind_grcs=[], 
                        ind_bigls=[], *args, **kwargs):
    """
    purpose is to compare the existing scales BIGLS & GRCS with those questions
    measuring the related latent factors
    input: 
        df (DataFrame) of whole questionaire
        ind_state (list) indices of the questions in the questionaire
        ind_grcs (list) indices of the questions ... GRCS
        ind_bigls (list) indices of the questions ... BIGLS

    output: none (just plot graph)
    """
    #Get matrices
    old_state_x = args[0]
    state_x = df.iloc[:, ind_state].as_matrix()
    grcs_x = df.iloc[:, ind_grcs].as_matrix()
    bigls_x = df.iloc[:, ind_bigls].as_matrix()
    
    #calculate scores
    old_state_y = score(old_state_x)
    state_y = score(state_x)
    grcs_y = score(grcs_x)
    bigls_y = score(bigls_x)
#    pdb.set_trace()
#    plot_conv(state_x, state_y, bigls_x, bigls_y)
    
    #BIGLS only covers luck and GRCS, State 2.0 and State 1.0 cover other
        #erroneous beliefs
    names = ['State', 'GRCS', 'Old State']
    x_plot = [state_x, grcs_x, old_state_x]
    y_plot = [state_y, bigls_y, grcs_y, old_state_y]
    
    if (kwargs.get("luck") == "yes"):
        names = ['State', 'BIGLS', 'GRCS', 'Old State']
        x_plot = [state_x, bigls_x, grcs_y, old_state_x]
        y_plot = [state_y, bigls_y, grcs_y, old_state_y]
        
    plot_conv(x_plot, y_plot, names,
              y_name=kwargs.get("y"), x_name=kwargs.get("x"),
              title=kwargs.get("title"))
    
    pdb.set_trace()
    return 0

def mean_graph(x, plot_std="yes"):
    """
    input: x as dataframe
    output: bar graph of mean with standard deviation
    """
    x = x.as_matrix()
    mean = x.mean(axis=0)
    std = np.std(x, axis=0)
    ticks = np.arange(len(mean))
    if (plot_std=="yes"):
        plt.bar(ticks, mean, yerr=std, align='center')
    if (plot_std=="no"):
        plt.bar(ticks, mean, align='center')
    plt.title("Mean of each question in State Con")
    plt.xlabel("Item of questionaire")
    plt.ylabel("Mean value")
    plt.tight_layout()
    plt.savefig(my_path+"/figures/response_mean.png")
    return 0

def drop_question(x, index):
    """Index = the number of the State_con scale question number or a list of 
    numbers
    x is the dataframe that only includes the State_con scale
    input: x (DataFrame)
    index (list of integers)
    
    output: x (DataFrame)
    """
    column_name = x.columns.values
    x = x.drop(columns=column_name[index],axis=1)
    return x

def bartlett_measure (x):
    """
    Perform the bartlett measure test on the State scale
    input: x matrix of answers as DataFrame
    output: test_statistic and p_value of homogeniety of the dataset
    
    CURRENTYL RETURN NAN VALUES (NAN, NAN)
    """
#    correlation =  x.corr().as_matrix()
    correlation = x.corr().as_matrix()
    bart = []
    for i in range(correlation.shape[0]-1):
        bart.append(correlation[i,i:][1:].tolist())
    test_statistic, p_val = ss.bartlett(*bart)
#    for i in range(correlation.shape[1]):
#        bart.append(correlation[:,i].tolist())
    return test_statistic, p_val


##########################Don't know if i still need it###########################################################
def efa_clean(x):
    """
    returns array of procedure checks (True or False)
    Array:
        determinant of correlation matrix larger than 0.00001
        KMO vaule
    """
    xcorr = x.corr(method="pearson")
    det_bool= np.linalg.det(xcorr) > 0.00001
#    bart_x = [x[:,i] for i in x]
    _, kmo_val = famissing.calculate_kmo(x)
    xcorr = np.abs(xcorr)
    #bartlett returns "T" which is the test statistic and p-value
    #perform the bartlett test on each individual input
#    _, bart_p = bartlett(*xcorr)
#    pdb.set_trace()
#    return [det_bool, kmo_val, bart_p]
    return [det_bool, kmo_val]

def scree_test(fa, kmo_data, num_factors):
    """
    fa is the factor analysis object of the package that contains eigenvalues
    kmo_data is the original matrix (not correlation)
    
    returns the anti-image matrix and kmo_Score
    """
    #ev is  original eigenvalues and v is common factor eigenvalues
    ev, v = fa.get_eigenvalues()
    plt.figure(4)
    plt.plot (ev)
    #calcualte Kaiser-Meyer-Olkin score to test sample size of data to be
    #   adequate or not
    #   float('%.1g' % score_total) just gives 1 decimal sig fig
    anti_image_matrix, score_total = famissing.calculate_kmo(kmo_data)
    plt.title("Kaiser-Myer-Olkin Criterion (KMO) score is... " + str(
            float('%.2g' % score_total)))
    plt.ylabel('eigenvalues')
    plt.xlabel(r'$\lambda_{x}$')
    plt.xticks(np.arange(num_factors))
    max_eig = max(ev.as_matrix())[0]
    plt.yticks(np.arange(max_eig, step=0.5))
    plt.axhline(y=1, color='r', linestyle='-')
    plt.savefig(my_path + "/figures/scree_test.png")
    plt.show()
    return anti_image_matrix, score_total


def trim (df, threshold):
    """Makes the dataframe or numpy matrix values 0 if below threshold
    input x is whatever matrix needs to be trimmed
    """
    x = df.copy()
    x[np.abs(x)<threshold] = 0
    return x

def efa (x, factors=3):
    fa = FactorAnalyzer()
    fa.analyze(x, n_factors=factors, rotation="quartimin")
    return fa

def cronbach_alpha(items):
    """
    adapted from 
    https://github.com/anthropedia/tci-stats/blob/master/tcistats/__init__.py
    
    Used as a measure of internal validity
    """
#    items = pandas.DataFrame(items)
    items_count = items.shape[1]
    variance_sum = float(items.var(axis=0, ddof=1).sum())
    total_var = float(items.sum(axis=1).var(ddof=1))
    
    return (items_count / float(items_count - 1) *
            (1 - variance_sum / total_var))

def report_participant_stats(read, columns):
    """
    input:
        read is a dataframe
        columns is the name of the column values in order of...
            Sex, Ethnicity, Age, Years spoking English, ...
    Report: Gender Plot
    Regional differences (Ethnicity)
    Age
    Years speaking english
    """
    
    """Sex"""
    sex = read[columns[0]].as_matrix()
    sex, sex_counts = np.unique(sex, return_counts=True)
    ticks = np.arange(len(sex))
    plt.figure(0)
    plt.bar(ticks, sex_counts)
    plt.xticks (ticks, ('Female', 'Male'))    
    plt.ylabel("Number of Participants")
    plt.title("Distribution of Female/Male Participants")
    plt.savefig(my_path + '/figures/sex_graph.png')
    
#    pdb.set_trace()
    """Ethnicity"""
    eth = read[columns[1]].as_matrix()
    eth, eth_counts = np.unique(eth, return_counts=True)
    ticks = np.arange(len(eth))
    plt.figure(1, figsize=(13,5))
    plt.bar(ticks, eth_counts)
    plt.xticks(ticks, eth)
    plt.ylabel("Number of Participants")
    plt.title("Ethnicity Distribution of Participant Pool")
    plt.savefig(my_path + '/figures/ethnicity.png')
    
    """Age"""
    age = read[columns[2]].as_matrix()
    age, age_counts = np.unique(age, return_counts=True)
    ticks = np.arange(len(age))
    plt.figure(2)
    plt.bar(ticks, age_counts)
    plt.xticks(ticks, age)
    plt.title("Age distribution of Pilot Data")
    plt.xlabel("Reported Age")
    plt.ylabel("Number of Participants")
    plt.savefig(my_path + "/figures/age_dist.png")
    
    #Years_Eng
    """Years Spoken in English"""
    eng_years = read[columns[3]].as_matrix()
    eng_years, eng_count = np.unique (eng_years, return_counts=True)
    ticks = np.arange(len(eng_years))
    plt.figure(3)
    plt.bar(ticks, eng_count)
    plt.xticks(ticks, eng_years)
    plt.title("Distribution of number of years spoken in English")
    plt.xlabel("Number of years")
    plt.ylabel("Number of Participants")
    plt.show()