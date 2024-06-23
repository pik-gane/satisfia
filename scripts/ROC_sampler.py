#This code creates the ROC Curve for testing the performance of models over different aspiration intervals
#Input of the environments' reward per episode with each aspiration is required as input
import numpy as np
import random
import matplotlib.pyplot as plt
class ROC_Curve:
    def __init__(self, min_value, max_value, no_bins):
        self.min = min_value
        self.max = max_value
        self.bins = no_bins
    
    def get_widths(self):
        max_width = self.max - self.min
        return np.linspace(0, max_width, self.bins)
    
    def get_intervals(self):
        intervals = []
        for i in self.get_widths():
            j = i/2
            centre_max = int(self.max - j)
            centre_min = int(self.min + j)
            centre = random.randint(centre_min,centre_max)
            intervals.append([int(centre-j),int(centre+j)])
        return list(reversed(intervals))
    
    def get_success(self, interval, env='PLACEHOLDER'):
        outcome = random.random() #To be replaced with the outcome of the environment
        if outcome >=0.5: #To be replaced
        #if outcome >=interval[0] and outcome <=interval[1]:
            return 1
        return 0

    def get_success_rates(self, env, iterations=1):
        success_frequency = {}
        for i in range(1,self.bins+1):
            success_frequency[i] = 0
        j=1
        while j <= iterations: 
            i=1
            intervals = self.get_intervals()
            while i<=self.bins:
                success_frequency[i] += self.get_success(env,intervals[i-1])
                i+=1
            j+=1
        for i in success_frequency.keys():
            success_frequency[i] = success_frequency[i]/iterations
        return success_frequency
    
    def plot_success_rates(self, env, bins, iterations=1):
        plt.plot(self.get_success_rates(env, iterations).values())
        plt.ylabel('ROC Curve')
        plt.show()
            
test = ROC_Curve(4, 400, 20)
test.plot_success_rates("PLACEHOLDER", 10, 1)