% Author: John Connor Prikkel
% Date: 9/14/23
% ECE 595 Assignment 3

function g = compute_sigmoid(z)
% Takes in an array/matrix as input and returns
% its sigmoid values
    g = 1./(1+exp(-z));
end