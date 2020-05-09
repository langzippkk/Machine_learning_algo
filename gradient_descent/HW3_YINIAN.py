import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline
from sklearn.linear_model import LogisticRegression
mpl.rc('figure', figsize=[10,6])

######################## Question 4###################################
def gradient_descent(X,y,theta,alpha = 0.01,lambda1 = 10,num_iters=10):
    ## cost function
    def cost_func(X,y,theta,lamda = 10):
        h = X @ theta
        h = 1./(1+np.exp(-h))
        m = len(y) 
    #    reg = (lamda / (2*m)) * np.sum(np.square(theta))
    #     res = float((1./(2*m)) * (h - y).T @ (h - y)) +reg
        reg = lamda* np.sum(np.square(theta)) 
        res = -(y.T@np.log(h)+(1-y).T@np.log(1-h))+reg
        res = np.squeeze(res)
        return(res) 

    std =  np.std(X, axis=0)
    mean =  np.mean(X, axis=0)
    X = (X - mean) / std
    X = np.c_[np.ones(X.shape[0]), X]
    cost = []
    m = np.size(y)
    plot_x = []
    for i in range(num_iters):
        if i>1 and abs(cost[i-1]-cost[i-2]) < 0.001:
            break
        h = np.dot(X,theta)
        h = 1/(1+np.exp(-h))
        ##theta = theta - alpha * (1/m)* ((X.T @ (h-y)) + lambda1 * theta)
        theta = np.array(theta - alpha * ((X.T @ (h-y)) + 2*lambda1 * theta))
        cost.append(cost_func(X,y,theta,lambda1))
        if cost[i] =='inf':
            print("learning rate is too high")
        plot_x.append(i)
        ## cost for plot
    plt.plot(plot_x,cost)
    plt.title('cost vs iterations')
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.show()
    ## denormalize
    cons = theta[0]
    theta = theta[1:]
    cons = (cons + np.sum([float(theta[i]*mean[i]/std[i]) for i in range(len(theta))])).reshape([1,])
    theta_other = [float(theta[i]/std[i]) for i in range(len(theta))]
    res = np.concatenate((cons, theta_other), axis=0)
    return res, cost


###############################Question 5 #######################################################
def read_data():
    df = pd.read_csv('wdbc.data', header=None)
    base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                     'conpoints', 'symmetry', 'fracdim']
    names = ['m' + name for name in base_names]
    names += ['s' + name for name in base_names]
    names += ['e' + name for name in base_names]
    names = ['id', 'class'] + names
    df.columns = names
    df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
    y = df['color']
    X =  np.array(df.drop(['id', 'color','class'], axis=1))
    return X,y,df
X,y,df = read_data()
def run_Question5(X,y):
    '''
    Inputs are X: the training data without predicted variable
    y: predicted variable
    '''
    ## initialize theta
    theta = np.array([1/len(X)]*(X.shape[1]+1))
    theta,cost = gradient_descent(X,y,theta,alpha = 0.001,lambda1 = 100,num_iters=10000)
    return theta

run_Question5(X,y)

######################## Question 6#########################################
## Question 6
def run_Question6(X,y):
    '''
    Inputs are X: the training data without predicted variable
    y: predicted variable
    '''
    ## normalize
    GD_Res = run_Question5(X,y)
    std =  np.std(X, axis=0)
    mean =  np.mean(X, axis=0)
    X = (X - mean) / std
    X = np.c_[np.ones(X.shape[0]), X]
    ## fitting
    y = y.astype(int)
    clf = LogisticRegression(random_state = 1,penalty='l2',C=0.01,max_iter=10000,solver='liblinear')
    res = clf.fit(X,y)
    theta = res.coef_[0]
    ## denormalize
    cons = theta[0]
    theta = theta[1:]
    cons = (cons + np.sum([float(theta[i]*mean[i]/std[i]) for i in range(len(theta))])).reshape([1,])
    theta_other = [float(theta[i]/std[i]) for i in range(len(theta))]
    output = np.concatenate((cons, theta_other), axis=0)
    ##### MAKE THE PLOT
    plt.plot(GD_Res,color='g')
    plt.title('gradient descent coefficient')
    plt.show()
    plt.plot(output,color='r')
    plt.title('sklearn coefficient')
    plt.show()
    print("sklearn theta output is :")
    print(output)
    print("my gradient descent theta output is ")
    print(GD_Res)
    return output

run_Question6(X,y)

###################### Question 7 #####################################
def run_Question7(df):
    '''
    this funciton take the whole dataframe
    '''
    my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')
    c1 = 'mradius'
    c2 = 'mtexture'
    clf = LogisticRegression(random_state = 11,penalty='l2',C=0.005,max_iter=10000,solver='lbfgs')
    X = df
    X['second_mradius'] = df[c1]**2
    X['third_mradius'] = df[c1]**3
    X['second_mtexture'] = df[c2]**2
    X['third_mtexture'] = df[c2]**3
    X['interaction1'] = df[c1]*df[c2]**2
    X['interaction2'] = df[c2]*df[c1]**2
     
    res = clf.fit(X[[c1, c2,'second_mradius','third_mradius','second_mtexture','third_mtexture','interaction1','interaction2']], df['color'])
    res.coef_
    plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
    plt.xlabel(c1)
    plt.ylabel(c2)

    x = np.linspace(df[c1].min(), df[c1].max(), 1000)
    y = np.linspace(df[c2].min(), df[c2].max(), 1000)
    xx, yy = np.meshgrid(x,y)
    second_mradius = xx.ravel().reshape(-1,1)**2
    third_mradius = xx.ravel().reshape(-1,1)**3
    second_mtexture = yy.ravel().reshape(-1,1)**2
    third_mtexture = yy.ravel().reshape(-1,1)**3
    interaction1 = xx.ravel().reshape(-1,1)* yy.ravel().reshape(-1,1)**2
    interaction2 = yy.ravel().reshape(-1,1)* xx.ravel().reshape(-1,1)**2
    predicted_prob = clf.predict_proba(np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1), second_mradius,third_mradius,second_mtexture,third_mtexture,interaction1,interaction2)))[:,1]
    predicted_prob = predicted_prob.reshape(xx.shape)
    plt.contour(xx, yy, predicted_prob, [0.5], colors=['b'])
    plt.show()

run_Question7(df)