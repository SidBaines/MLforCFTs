#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.random import check_random_state
from sympy import sympify, lambdify
from datetime import datetime
import os


# In[76]:


saveDir = "output/" + datetime.now().strftime("%Y%m%d-%H%M%S")  + "_SymbolicRegression_2d_Example/"
os.makedirs(saveDir)
print(saveDir)


# In[93]:


# Cell to generate x and y values.
# Here, we define the function, the x-range and the number of points, then calculate the y values, then plot them
# nb I couldn't work out how to make the lambda functions take in an undetermined number of X dimensions, so for the moment it's hardcoded and can only take values up to 5-dimensional input.
# Also we can only plot when the number of X dimensions is 1 or 2 (for obvious reasons)
if 0:
    N_DIMS = 1 # Number of X dimensions
    X_MIN = [-1] # List of length N_DIMS, containing for each dimension the minimum of the X range to be used for training
    X_MAX = [1] # List of length N_DIMS, containing for each dimension the maximum of the X range to be used for training
    def my_func(x): # The function which we are trying to guess. Takes in an (:, N_DIMS) np array and returns a (:, 1) np array with the function applied
        return np.exp(x) 
    eqns = {'Ground Truth' : 'exp(x)'} # Just used for labelling plots, if you want to change the funciton you have to change the definition of my_func (and then preferably update this so the plots are labelled correctly)
elif 0:
    N_DIMS = 1
    X_MIN = [-1]
    X_MAX = [1]
    eqns = {'Ground Truth' : 'sin(3x)'} # This is just for labelling on plots; should be changed to match whatever we make 'my_func' below
    def my_func(x):
        return np.sin(3*x)
elif 0:
    N_DIMS = 1
    X_MIN = [-1]
    X_MAX = [1]
    eqns = {'Ground Truth' : 'sin(3x) - 2*sin(x)'} # This is just for labelling on plots; should be changed to match whatever we make 'my_func' below
    def my_func(x):
        return np.sin(3*x) - 2*np.sin(x)
elif 0:
    N_DIMS = 1
    X_MIN = [-10]
    X_MAX = [10]
    eqns = {'Ground Truth' : 'sin(0.311x) - pi*sin(x) + 1.3*cos(5.2*x)'} # This is just for labelling on plots; should be changed to match whatever we make 'my_func' below
    def my_func(x):
        return np.sin(0.311*x) - np.pi*np.sin(x) + 1.3*np.cos(5.2*x)
elif 1:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : 'sin(0.311*x0) - pi*sin(x1) + 1.3*cos(5.2*x0 - 3.5*x1)'} # This is just for labelling on plots; should be changed to match whatever we make 'my_func' below
    def my_func(x):
        return np.sin(0.311*x[:,0]) - np.pi*np.sin(x[:,1]) + 1.3*np.cos(5.2*x[:,0]-3.5*x[:,1])
elif 1:
    N_DIMS = 1
    X_MIN = [-1]
    X_MAX = [1]
    eqns = {'Ground Truth' : 'sin(2*x) - 2*sin(5*x) + 4'}
    def my_func(x):
        return np.sin(2*x) - 2*np.sin(5*x) + 4
        # return np.exp(1*x) - x + 2
elif 1:
    N_DIMS = 1
    X_MIN = [-1]
    X_MAX = [1]
    eqns = {'Ground Truth' : 'exp(x) - x + 2'}
    def my_func(x):
        return np.exp(1*x) - x + 2
elif 0:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : 'exp(x0 + x1)'}
    def my_func(x):
        return np.exp(x[:,0] + x[:,1])
elif 0:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : 'sin(x0 + x1) - cos(x1)'}
    def my_func(x):
        return np.sin(x[:,0] + x[:,1]) - np.cos(x[:,1])
elif 0:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : 'x0^2 - x1^2 + x1 -1'}
    def my_func(x):
        return x[:,0]*x[:,0] - x[:,1]*x[:,1] + x[:,1] - 1
elif 0:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : 'sin(x0 - x1)'}
    def my_func(x):
        return np.sin( x[:,0] - x[:,1] )
elif 0:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : '3*sin(x0 - x1) - 2*cos(x1 - x0)'}
    def my_func(x):
        return 3*np.sin( x[:,0] - x[:,1] ) - 2*np.cos( x[:,1] - x[:,0] )
elif 0:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : '1.3*sin(x0 + x1) - 0.2*cos(x1 - x0)'}
    def my_func(x):
        return 1.3*np.sin( x[:,0] + x[:,1] ) - 0.2*np.cos( x[:,1] - x[:,0] )
elif 0: # Bad example
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : '1.3*sin(x0 + x1) - 0.2*cos(x1 - x0) +  5.2*cos(x1/x0)'}
    def my_func(x):
        return 1.3*np.sin( x[:,0] + x[:,1] ) - 0.2*np.cos( x[:,1] - x[:,0] + 5.2*np.cos(x[:,1]/x[:,0]))
elif 1:
    N_DIMS = 2
    X_MIN = [-1, -1]
    X_MAX = [1, 1]
    eqns = {'Ground Truth' : '0.4*sin(3*x0-0.2*x1) - 0.82*cos(x0+2.43*x1) - cos(4*x0)'}
    def my_func(x):
        return 0.4*np.sin(3*x[:,0]-0.2*x[:,1]) - 0.82*np.cos(x[:,0]+2.43*x[:,1]) - np.cos(4*x[:,0])
else:
    assert(0)

if N_DIMS == 1:
    N_POINTS = 500
    extension_factor = 1 # The factor by which we extend the range when we do the extended range test. Factor == 1 will extend the range by 50% of the original range to the left and right each. Factor 2 will do 100% to left and right, factor 0.5 will do 25% each left and right, etc.
    x0 = np.arange(X_MIN[0], X_MAX[0], (X_MAX[0] - X_MIN[0])/N_POINTS) # Core region, on which we will train and test
    x0_ext = np.arange(X_MIN[0] - (X_MAX[0]-X_MIN[0])/2, X_MAX[0] + (X_MAX[0]-X_MIN[0])/2, (X_MAX[0]-X_MIN[0])/N_POINTS) # Region slightly below (x-range half the size of the core region) on which to see if we extrapolate well
    y_truth = my_func(x0)
    y_truth_ext = my_func(x0_ext)

    ax = plt.figure().add_subplot()
    # ax.set_xlim(-1, 1)
    ax.set_xlim(X_MIN[0] - (X_MAX[0]-X_MIN[0])/2, X_MAX[0] + (X_MAX[0]-X_MIN[0])/2)
    # ax.set_ylim(-1, 1)
    ax.plot(x0_ext, y_truth_ext, color='green', alpha=0.2)
    ax.plot(x0, y_truth, color='red', alpha=0.5)
    # plt.yscale('log')
    plt.show()
elif N_DIMS == 2:
    # The remainder of this cell is just for plotting; the data here is NOT used for training; it is an evenly spaced grid
    N_POINTS = 100
    x0 = np.arange(X_MIN[0], X_MAX[0], (X_MAX[0] - X_MIN[0])/N_POINTS) # Core region, on which we will train and test
    x1 = np.arange(X_MIN[1], X_MAX[1], (X_MAX[1] - X_MIN[1])/N_POINTS) # Core region, on which we will train and test
    x0, x1 = np.meshgrid(x0, x1)
    x_comb=np.concatenate((x0.reshape(-1,1), x1.reshape(-1,1)), axis=1)
    x0_ext = np.arange(X_MIN[0] - (X_MAX[0]-X_MIN[0])/2, X_MAX[0] + (X_MAX[0]-X_MIN[0])/2, (X_MAX[0]-X_MIN[0])/N_POINTS) # Region slightly below (x-range half the size of the core region) on which to see if we extrapolate well
    x1_ext = np.arange(X_MIN[1] - (X_MAX[1]-X_MIN[1])/2, X_MAX[1] + (X_MAX[1]-X_MIN[1])/2, (X_MAX[1]-X_MIN[1])/N_POINTS) # Region slightly below (x-range half the size of the core region) on which to see if we extrapolate well
    x0_ext, x1_ext = np.meshgrid(x0_ext, x1_ext)
    x_comb_ext=np.concatenate((x0_ext.reshape(-1,1), x1_ext.reshape(-1,1)), axis=1)
    y_truth = my_func(x_comb)
    y_truth = y_truth.reshape(x0.shape)
    y_truth_ext = my_func(x_comb_ext)
    y_truth_ext = y_truth_ext.reshape(x0_ext.shape)

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(X_MIN[0] - (X_MAX[0]-X_MIN[0])/2, X_MAX[0] + (X_MAX[0]-X_MIN[0])/2)
    ax.set_ylim(X_MIN[1] - (X_MAX[1]-X_MIN[1])/2, X_MAX[1] + (X_MAX[1]-X_MIN[1])/2)
    surf = ax.plot_surface(x0_ext, x1_ext, y_truth_ext, rstride=1, cstride=1,
                        color='green', alpha=0.2)
    surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1,
                        color='red', alpha=0.5)
    # plt.show()


# In[94]:


# Cell to generate train and test data within the region, at randomly sampled points
rng = check_random_state(0)
N_TRAIN=1000
N_TEST=1000

# Training samples
X_train = np.concatenate(list(rng.uniform(X_MIN[i], X_MAX[i], N_TRAIN).reshape(-1,1) for i in range(N_DIMS)), axis=1)
y_train = my_func(X_train).reshape(-1,1)

# Testing samples
X_test = np.concatenate(list(rng.uniform(X_MIN[i], X_MAX[i], N_TEST).reshape(-1,1) for i in range(N_DIMS)), axis=1)
y_test = my_func(X_test).reshape(-1,1)

X_test_ext = np.concatenate(list(rng.uniform(X_MIN[i] - (X_MAX[i]-X_MIN[i])/2, X_MAX[i] + (X_MAX[i]-X_MIN[i])/2, N_TEST).reshape(-1,1) for i in range(N_DIMS)), axis=1)
y_test_ext = my_func(X_test_ext).reshape(-1,1)


# In[95]:


converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y
}


# In[96]:


# Cell to define a class used as a wrapper such that we can use a symbolic expression to make predictions
from sklearn.base import RegressorMixin
class MyEstimator(RegressorMixin):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def predict(self, X):
        # Check for constancy
        if X.shape[1] == 1:
            vals = self.func(X[:,0])
        elif X.shape[1] == 2:
            vals = self.func(X[:,0], X[:,1])
        elif X.shape[1] == 3:
            vals = self.func(X[:,0], X[:,1], X[:,2])
        elif X.shape[1] == 4:
            vals = self.func(X[:,0], X[:,1], X[:,2], X[:,3])
        elif X.shape[1] == 5:
            vals = self.func(X[:,0], X[:,1], X[:,2], X[:,3], X[:,4])
        else:
            assert(0) # For now I couldn't work out how to get lambdify to work such that we can give it a tuple or a list of inputs, or to give an unnknown number of inputs to a python function, so I am just hardcoding up to 5d.
        try: # if the function is constant, then 'len' will fail as it'll come out as np.float which has no len
            len(vals)
            return vals
        except:
            return np.ones(X.shape)*vals


# In[97]:


# Cell to approximate using aifeynman package. 
# This one is hard to use in terms of input/output; it's not written to use scikit-learn-like syntax (or anything even remotely close)
# Nevertheless, we try it
if 0:
    import aifeynman
    data = np.concatenate((X_train,y_train), axis=1)
    np.savetxt(saveDir + 'mystery_train.txt', data)
    aifeynman.run_aifeynman(saveDir,"mystery_train.txt",20,"14ops.txt", polyfit_deg=3, NN_epochs=100)
    all_eqs={}
    with open('results/solution_mystery_train.txt') as f:
        l = f.readline()
        while(len(l)>0):
            all_eqs[float(l.split(' ')[0])] = l.split(' ')[-1].replace('\n','')
            l = f.readline()
    all_eqs
    best_eq = all_eqs[min(list(all_eqs.keys()))]
    print(sympify(best_eq))

    all_inp_syms = 'x0'
    for i in range(1, N_DIMS):
        all_inp_syms += ', x%d' %(i)
    lam_f = lambdify(all_inp_syms, sympify(best_eq))
    eqns['AiFeynSymbolicRegressor'] = sympify(best_eq, locals=converter)
    est_aifeyn = MyEstimator(lam_f)


# In[98]:


# Cell to fit using gplearn
# This one struggled on some pretty easy 1d examples; it's likely that I just wasn't getting the parameters/setup right 
# but it wasn't super intuitive and I found it much easier to achieve success with some of the others. It also broke when 
# I tried adding the exponential as a custom function.
# However, does have easy ability to add own functions, which could be really helpful for us
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

def exponent(x):
  a = np.exp(x)
  a[~np.isfinite(a)] = 0
  return a
new_exp = make_function(function=exponent,
                        name='exp',
                        arity=1)
est_gp = SymbolicRegressor(population_size=5000,
                           generations=200, stopping_criteria=0.001,
                           p_crossover=0.5, p_subtree_mutation=0.15,
                           p_hoist_mutation=0.15, p_point_mutation=0.15,
                           p_point_replace=0.2,
                           max_samples=0.9, verbose=1, random_state=0,
                           const_range=(-10,10),
                          #  init_depth=((3,8)),
                          #  init_method='half and half',
                          #  metric='mean absolute error',
                        #    parsimony_coefficient=0.02,
                           # parsimony_coefficient='auto',
                           parsimony_coefficient=0.001,
                           tournament_size=10,
                          #  function_set=['add', 'mul', 'sin', 'cos', 'sub', new_exp],
                           function_set=['add', 'sub', 'mul', 'sin', 'cos', 'inv'],
                           )
est_gp.fit(X_train, y_train.ravel())
print(est_gp._program)
print(sympify(str(est_gp._program), locals=converter))
sympify(str(est_gp._program), locals=converter)

eqns['GPlearnSymbolicRegressor'] = sympify(str(est_gp._program), locals=converter)


# In[99]:


import graphviz
print(est_gp._program.parents)
dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph


# In[100]:


# Cell to fit using DecisionTreeRegressor. This won't be able to extend beyone the range that we fit on, 
# but a good comparison to see how well we can do on the training range
from sklearn.tree import DecisionTreeRegressor
est_tree = DecisionTreeRegressor()
est_tree.fit(X_train, y_train)
eqns['DecisionTreeRegressor'] = None


# In[101]:


# Cell to fit using RandomForestRegressor. This won't be able to extend beyone the range that we fit on, 
# but a good comparison to see how well we can do on the training range
from sklearn.ensemble import RandomForestRegressor
est_rf = RandomForestRegressor(n_estimators=10)
est_rf.fit(X_train, y_train.reshape((-1,)))
eqns['RandomForestRegressor'] = None


# In[102]:


import feyn
import pandas as pd
train = pd.DataFrame(np.concatenate((X_train,y_train), axis=1), columns=['X%d' %(i) for i in range(N_DIMS)] + ['y'])
ql = feyn.QLattice()
models = ql.auto_run(train, output_name = 'y',
                     n_epochs=25,
                     )
print(models[0].sympify(signif=3))
all_inp_syms = 'X0'
for i in range(1, N_DIMS):
    all_inp_syms += ', X%d' %(i)
est_ql=MyEstimator(lambdify(all_inp_syms,models[0].sympify(signif=3)))
eqns['QuantumLatticeSymbolicRegressor'] = models[0].sympify(signif=3)


# In[108]:


if 1:
    # Cell to fit using PySr, another Symbolic regression package. This one is able to do the few simple 1d 
    # examples I've tried, and also is said to do well in the rankings from the SRBench competition 2022 
    # (https://cavalab.org/srbench/competition-2022/). Seems to have syntax for adding custom functions 
    # (though I have not yet tested this).
    from pysr import PySRRegressor
    est_pysr = PySRRegressor(
        # niterations=40,  # < Increase me for better results
        niterations=80,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            # "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
        #  verbose=True,
        # procs=0, # Prevents multiprocessing (slower for long run jobs, but avoids expensive startup cost for short running jobs)
        populations=25,
        population_size=200,
        maxsize=30,
    )
    try:
        est_pysr.fit(X_train, y_train)
        eqns['PySrSymbolicRegressor'] = est_pysr.sympy()
        print(est_pysr.sympy())
        pysr_finished=True
    except KeyboardInterrupt:
        pysr_finished=False


# In[89]:


eqns['RandomForestRegressor'] = None
eqns['DecisionTreeRegressor'] = None
eqns['QuantumLatticeSymbolicRegressor'] = models[0].sympify(signif=3)
eqns['PySrSymbolicRegressor'] = est_pysr.sympy()


# In[104]:


if N_DIMS == 1:
    ys = {}
    scores = {}
    minx, miny = min(x0), min(y_truth)
    maxx, maxy = max(x0), max(y_truth)

    ys['Ground Truth'] = {'main':y_truth, 'ext':y_truth_ext}
    scores['Ground Truth'] = {'main':None, 'ext':None}
    all_predictors = {
        'GPlearnSymbolicRegressor' : est_gp, 
        'PySrSymbolicRegressor' : est_pysr, 
        'QuantumLatticeSymbolicRegressor' : est_ql, 
        # 'AiFeynSymbolicRegressor' : est_aifeyn,
        'DecisionTreeRegressor' : est_tree, 
        'RandomForestRegressor' : est_rf, 
    }
    for predictor_name in all_predictors.keys():
        predictor = all_predictors[predictor_name]
        ys[predictor_name] = {}
        scores[predictor_name] = {}
        for pred_set, test_set_x, test_set_y, set_name in zip([x0, x0_ext],
                                            [X_test, X_test_ext],
                                            [y_test, y_test_ext],
                                            ['main', 'ext']):
            ys[predictor_name][set_name] = predictor.predict(pred_set.reshape(-1,1)).reshape(pred_set.shape)
            try:
                scores[predictor_name][set_name] = predictor.score(test_set_x, test_set_y)
            except:
                scores[predictor_name][set_name] = None

    xs = {}
    xs['main'] = x0
    xs['ext'] = x0_ext

    fig = plt.figure(figsize=(12, 20))
    for i, title in enumerate(ys.keys()):
        y = ys[title]
        score = scores[title]
        # ax = fig.add_subplot(3, 2, i+1)
        ax = fig.add_subplot(6, 1, i+1)
        # ax.set_xlim(-1, 1)
        # ax.set_xlim(X_MIN - (X_MAX-X_MIN)/2, X_MAX + (X_MAX-X_MIN)/2)
        # ax.set_ylim(-1, 1)
        # ax.set_xticks(np.arange(-1, 1.01, .5))
        # ax.set_yticks(np.arange(-1, 1.01, .5))
        # surf = ax.plot(x0, y, color='green', alpha=0.5, label=title)
        for k in xs.keys():
            # ax.plot(xs[k], y, color='green'*(k=='main') + 'red'*(k!='main'), alpha=0.5, label=title)
            if k == 'main':
                ax.plot(xs[k], y[k], color='green', alpha=0.3 + (k=='main')*0.4, label=title)
            else:
                ax.plot(xs[k], y[k], color='green', alpha=0.3 + (k=='main')*0.4)
        n_p=100
        marker_s = 5
        ax.scatter(X_train[:n_p], y_train[:n_p], color='b', label='train points', s=marker_s)
        for k, test_set_x, test_set_y in zip(
            ['main', 'ext'],
            [X_test, X_test_ext],
            [y_test, y_test_ext],
        ):
            if k == 'main':
                ax.scatter(test_set_x[:n_p], test_set_y[:n_p], color='r', alpha=0.7, label='test points', s=marker_s)
            else:
                ax.scatter(test_set_x[:n_p], test_set_y[:n_p], color='r', alpha=0.3, s=marker_s)
        plt.legend()
        # plt.yscale('log')
        # plt.ylim([miny - 0.5*(maxy-miny), maxy + 0.5*(maxy-miny)])
        total_title = title
        if score['main'] is not None:
            # ax.text(minx, miny, "$R^2 =\/ %.6f (%.6f, %.6f)$" % (score['main'], score['ext_lower'], score['ext_upper']), fontsize=10)
            total_title += "\n$R^2 =\/ %.6f$  $(%.6f)$" % (score['main'], score['ext'])
        if eqns[title] is not None:
            total_title += "\n%s" %(eqns[title])

        plt.title(total_title)
    plt.tight_layout()
    plt.savefig(saveDir + 'AllEstimators_1d.pdf')
    # plt.show()
elif N_DIMS == 2:
    ys = {}
    scores = {}
    ys['Ground Truth'] = {'main':y_truth, 'ext':y_truth_ext}
    scores['Ground Truth'] = {'main':None, 'ext':None}
    all_predictors = {
        'GPlearnSymbolicRegressor' : est_gp, 
        'PySrSymbolicRegressor' : est_pysr, 
        'QuantumLatticeSymbolicRegressor' : est_ql, 
        # 'AiFeynSymbolicRegressor' : est_aifeyn,
        'DecisionTreeRegressor' : est_tree, 
        'RandomForestRegressor' : est_rf, 
    }
    if pysr_finished:
        all_predictors['PySrSymbolicRegressor'] = est_pysr
    for predictor_name in all_predictors.keys():
        predictor = all_predictors[predictor_name]
        ys[predictor_name] = {}
        scores[predictor_name] = {}
        for pred_set, pred_set_shaper, test_set_x, test_set_y, set_name in zip(
                                            [x_comb, x_comb_ext],
                                            [x0, x0_ext],
                                            [X_test, X_test_ext],
                                            [y_test, y_test_ext],
                                            ['main', 'ext']):
            # ys[predictor_name][set_name] = predictor.predict(pred_set.reshape(-1,1)).reshape(pred_set.shape)
            ys[predictor_name][set_name] = predictor.predict(pred_set).reshape(pred_set_shaper.shape)
            try: # Try making a score, might fail if some of the calculations are impossible
                scores[predictor_name][set_name] = predictor.score(test_set_x, test_set_y)
            except:
                scores[predictor_name][set_name] = None
    xs = {}
    xs['main'] = [x0, x1]
    xs['ext'] = [x0_ext, x1_ext]
    fig = plt.figure(figsize=(12, 20))
    for i, title in enumerate(ys.keys()):
        y = ys[title]
        score = scores[title]
        # ax = fig.add_subplot(3, 2, i+1)
        ax = fig.add_subplot(6, 1, i+1, projection='3d')
        for k in xs.keys():
            # ax.plot(xs[k], y, color='green'*(k=='main') + 'red'*(k!='main'), alpha=0.5, label=title)
            if k == 'main':
                ax.plot_surface(xs[k][0], xs[k][1], y[k], rstride=1, cstride=1, color='green', alpha=0.3 + (k=='main')*0.4, label=title)
            else:
                ax.plot_surface(xs[k][0], xs[k][1], y[k], rstride=1, cstride=1, color='green', alpha=0.3)
        n_p=100
        marker_s = 5
        ax.scatter(X_train[:n_p, 0], X_train[:n_p, 1], y_train[:n_p], color='b', label='train points', s=marker_s)
        for k, test_set_x, test_set_y in zip(
            ['main', 'ext'],
            [X_test, X_test_ext],
            [y_test, y_test_ext],
        ):
            if k == 'main':
                ax.scatter(test_set_x[:n_p, 0], test_set_x[:n_p, 1], test_set_y[:n_p], color='r', alpha=0.7, label='test points', s=marker_s)
            else:
                ax.scatter(test_set_x[:n_p, 0], test_set_x[:n_p, 1], test_set_y[:n_p], color='r', alpha=0.3, s=marker_s)
        plt.legend()
        # plt.yscale('log')
        # plt.ylim([miny - 0.5*(maxy-miny), maxy + 0.5*(maxy-miny)])
        total_title = title
        if score['main'] is not None:
            # ax.text(minx, miny, "$R^2 =\/ %.6f (%.6f, %.6f)$" % (score['main'], score['ext_lower'], score['ext_upper']), fontsize=10)
            # total_title += "\n$R^2 =\/ %.6f$  $(%.6f, %.6f)$" % (score['main'], score['ext_lower'], score['ext_upper'])
            total_title += ": $R^2 =\/ %.6f$" % (score['main'])
        if score['ext'] is not None:
            total_title += " ($R^2 =\/ %.6f$)" % (score['ext'])
        if eqns[title] is not None:
            total_title += "\n%s" %(eqns[title])
        plt.title(total_title)
    plt.tight_layout()
    plt.savefig(saveDir + 'AllEstimators_2d.pdf')
    # plt.show()







'''
if 0: # The above but just without comments or useless lines so that I can replot in an interactive python run more easily
ys = {}
scores = {}
ys['Ground Truth'] = {'main':y_truth, 'ext':y_truth_ext}
scores['Ground Truth'] = {'main':None, 'ext':None}
all_predictors = {
    'GPlearnSymbolicRegressor' : est_gp, 
    'PySrSymbolicRegressor' : est_pysr, 
    'QuantumLatticeSymbolicRegressor' : est_ql, 
    'DecisionTreeRegressor' : est_tree, 
    'RandomForestRegressor' : est_rf, 
}
if pysr_finished:
    all_predictors['PySrSymbolicRegressor'] = est_pysr
for predictor_name in all_predictors.keys():
    predictor = all_predictors[predictor_name]
    ys[predictor_name] = {}
    scores[predictor_name] = {}
    for pred_set, pred_set_shaper, test_set_x, test_set_y, set_name in zip(
                                        [x_comb, x_comb_ext],
                                        [x0, x0_ext],
                                        [X_test, X_test_ext],
                                        [y_test, y_test_ext],
                                        ['main', 'ext']):
        ys[predictor_name][set_name] = predictor.predict(pred_set).reshape(pred_set_shaper.shape)
        try: # Try making a score, might fail if some of the calculations are impossible
            scores[predictor_name][set_name] = predictor.score(test_set_x, test_set_y)
        except:
            scores[predictor_name][set_name] = None
xs = {}
xs['main'] = [x0, x1]
xs['ext'] = [x0_ext, x1_ext]
fig = plt.figure(figsize=(12, 20))
for i, title in enumerate(ys.keys()):
    y = ys[title]
    score = scores[title]
    ax = fig.add_subplot(6, 1, i+1, projection='3d')
    for k in xs.keys():
        if k == 'main':
            ax.plot_surface(xs[k][0], xs[k][1], y[k], rstride=1, cstride=1, color='green', alpha=0.3 + (k=='main')*0.4, label=title)
        else:
            ax.plot_surface(xs[k][0], xs[k][1], y[k], rstride=1, cstride=1, color='green', alpha=0.3)
    n_p=100
    marker_s = 5
    ax.scatter(X_train[:n_p, 0], X_train[:n_p, 1], y_train[:n_p], color='b', label='train points', s=marker_s)
    for k, test_set_x, test_set_y in zip(
        ['main', 'ext'],
        [X_test, X_test_ext],
        [y_test, y_test_ext],
    ):
        if k == 'main':
            ax.scatter(test_set_x[:n_p, 0], test_set_x[:n_p, 1], test_set_y[:n_p], color='r', alpha=0.7, label='test points', s=marker_s)
        else:
            ax.scatter(test_set_x[:n_p, 0], test_set_x[:n_p, 1], test_set_y[:n_p], color='r', alpha=0.3, s=marker_s)
    plt.legend()
    total_title = title
    if score['main'] is not None:
        total_title += ": $R^2 =\/ %.6f$" % (score['main'])
    if score['ext'] is not None:
        total_title += " ($R^2 =\/ %.6f$)" % (score['ext'])
    if eqns[title] is not None:
        total_title += "\n%s" %(eqns[title])
    plt.title(total_title)
plt.tight_layout()
plt.savefig(saveDir + 'AllEstimators_2d.pdf')
'''