function [obj,g] = testFunc(x,a,b)
obj = a*(b-x)^2;
g = -2*a*(b-x);