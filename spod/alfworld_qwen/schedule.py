"""Learning-rate schedule."""
import math

def learning_rate(u):
 assert 1<=u<=50
 return 1e-5*u/3 if u<=3 else 1e-6+9e-6*(1+math.cos(math.pi*(u-3)/47))/2
