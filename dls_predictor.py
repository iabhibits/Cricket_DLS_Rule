#!/usr/bin/env python
# coding: utf-8
# author: Abhishek Kumar


import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_data(filename):
    df = pd.read_csv(filename)
    df.head()
    print(df.shape)
    data = df[df['Innings']==1]
    #data.head()
    #data.info()
    #print(data.shape)
    return data

def get_over_run(data):

    arr = []
    for j in range(1,11):
        d = data[data['Wickets.in.Hand'] == j]
        over = d['Total.Overs'].values - d['Over'].values
        run = d['Innings.Total.Runs'].values - d['Total.Runs'].values
        for i in range(d.shape[0]):
            x = [over[i],run[i],j]
            arr.append(x)
    run = data['Innings.Total.Runs'].values
    over = data['Total.Overs'].values
    print(run)
    uniqueValues, indicesList = np.unique(run, return_index=True)
    #print(uniqueValues)
    #print(indicesList)
    for i in indicesList:
        arr.append([over[i],run[i],10])
    return np.array(arr)

def cal_run(z,l,u):
    run = z * (1 - np.exp(-l*u/z))
    return run

def cal_loss(z,arr):
    #print(z)
    loss = 0
    for i in range(arr.shape[0]):
        prun = cal_run(z[arr[i][2]],z[10],arr[i][0])
        loss += (prun-arr[i][1]) ** 2
    return loss

def minimize_loss(arr):
    #z = np.sort(random.sample(range(1, 300),10))
    z = [10.0, 30.0, 60.0, 100.0, 140.0, 180.0, 220.0, 250.0, 270.0, 280.0, 4]
    #l = random.randint(1,5)
    #z = np.append(z,l)
    #make constraints such that z0 <= z1 <= z2 ... <= z9
    cons = (
    {'type': 'ineq','fun': lambda z: z[1] - z[0]},
    {'type': 'ineq','fun': lambda z: z[2] - z[1]},
    {'type': 'ineq','fun': lambda z: z[3] - z[2]},
    {'type': 'ineq','fun': lambda z: z[4] - z[3]},
    {'type': 'ineq','fun': lambda z: z[5] - z[4]},
    {'type': 'ineq','fun': lambda z: z[6] - z[5]},
    {'type': 'ineq','fun': lambda z: z[7] - z[6]},
    {'type': 'ineq','fun': lambda z: z[8] - z[7]},
    {'type': 'ineq','fun': lambda z: z[9] - z[8]},
    {'type': 'ineq','fun': lambda z: z[10] - z[0]},
    {'type': 'ineq','fun': lambda z: z[1] - z[10]})
    #parameter = minimize(cal_loss,z,args = arr,constraints= cons)
    parameter = minimize(cal_loss,z,args = arr,method = 'L-BFGS-B')
    #parameter = minimize(cal_loss,z,args = arr,method = 'TNC')
    #print(parameter)
    opt_param = parameter['x']
    loss = parameter['fun']
    return opt_param,loss


def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    return x,y

def plots(opt_param):
    colororder=[(0.00,0.00,1.00),
                (0.00,0.50,0.00),
                (1.00,0.00,0.00),
                (0.00,0.75,0.75),
                (0.75,0.00,0.75),
                (0.75,0.75,0.00),
                (0.25,0.25,0.25),
                (0.75,0.25,0.25),
                (0.95,0.95,0.00),
                (0.00,1.00,0.00),
                (0.76,0.57,0.17),
                (0.25,0.25,0.75)]
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize = (5,5),dpi = 100)
    ax.set_title('DLS_Resource_Calculator_Chart'.format('seaborn'))
    plt.xticks([i*5 for i in range(0,11)])
    plt.yticks([i*10 for i in range(0,11)])
    plt.xlabel('Overs remaining')
    plt.ylabel('Percentage of resource remaining')
    max_run = cal_run(opt_param[9], opt_param[10], 50)
    x = np.arange(50,-1,-1)
    plot_x = np.arange(0,51)
    for i in range(10):
        plot_y = 100 * cal_run(opt_param[i], opt_param[10], x)/max_run
        ax.plot(plot_x,plot_y,c =colororder[i],
                linewidth = 1,label = 'Z( '+str(i)+' )'.format('seaborn'))
    x,y = graph('(x * -2) + 100', range(0,51))
    plt.plot(x,y,linewidth = 0.9,c = colororder[11])
    ax.legend()
    fig.savefig('DLS_Curve.png')
    plt.show()

if __name__ == '__main__':
    data = get_data('04_cricket_1999to2011.csv')
    arr = get_over_run(data)
    z,loss = minimize_loss(arr)
    print(z,loss)
    plots(z)


