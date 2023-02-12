clc; clear; close all

% load data
load('./tests/Hubble.mat');
D = M;
p = 0.9;
M = rand(size(D))<=p;
r = 8;
[W,H] = masked_nnmf(D,M,r, 'init_mode', 'kmeans', 'maxiter', 500);
MD = M.*D;
D_hat = W*H;
showMatrixColumns(MD(:,1:10),10,128,128);
showMatrixColumns(D(:,1:10),10,128,128);
showMatrixColumns(D_hat(:,1:10),10,128,128);

