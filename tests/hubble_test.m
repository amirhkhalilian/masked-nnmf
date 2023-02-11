clc; clear; close all

% load data
load('./tests/Hubble.mat');
D = M;
M = ones(size(D));
r = 8;
[W,H] = masked_nnmf(D,M,r, 'init_mode', 'nnmf', 'maxiter', 500);
showMatrixColumns(W,r,128,128);

[W_m, H_m] = nnmf(D,r);
showMatrixColumns(W_m,r,128,128);

