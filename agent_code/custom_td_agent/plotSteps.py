import matplotlib.pyplot as plt
import numpy as np


def plotStepsHist(steps):      
    #plt.xlim((0,500))
    plt.hist(steps[1:], density=True)
    plt.xlabel("Steps per round")
    plt.savefig('stepsHist.png')

def plotSteps(steps):
    plt.plot(steps)
    plt.xlabel('Game')
    plt.ylabel('Steps')
    plt.savefig('steps.png')



steps = np.loadtxt('steps.txt')
#plotSteps(steps)
plotStepsHist(steps)

