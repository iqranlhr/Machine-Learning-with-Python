import numpy as np

x = np.arange(10, 200, 3)
y = np.arange(10, 200, 3) * 2


class SimpleLinearRegression:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        self.slop = self.calculate_slop()
        self.intercept = self.catculate_intercept()
       
    def catculate_intercept(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        return y_mean - self.slop * x_mean

    def calculate_slop(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        numinator = sum(abs(x-x_mean) * abs(y-y_mean) for x, y in zip(self.x, self.y))
        denominator = sum(abs(x-x_mean)**2 for x in self.x )
        return abs(numinator / denominator)

    def prediction(self, *args):
        #y = mx + c
        return [self.intercept + self.slop * i for i in args]
    
    def error(self, actual_value, predicted_values):
        return sum(abs(a - p) for a,p in zip(actual_value, predicted_values)) / len(predicted_values)

# Testing
# obj = SimpleLinearRegression(x, y)
# print(obj.slop)
# print(obj.intercept)
# print(obj.prediction(10)[0])
# print(x[0])
# print(y[0])
# checking error
# print(obj.error([y[0]], [obj.prediction(10)[0]]))
