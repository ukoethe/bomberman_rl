import matplotlib.pyplot as plt
import numpy as np


def plotScoresHist(scores):      
    plt.hist(scores[1:], density=True)
    plt.xlabel("ScorePerRound")
    plt.savefig('scoresHist.png')

def plotScores(scores):
    plt.plot(scores)
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.savefig('scores.png')


scores = np.loadtxt('scores.txt')
plotScoresHist(scores)
#plotScores(scores)

