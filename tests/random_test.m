clc; clear; close all;

%% Create sample data

m = 100; % number of rows
n = 10000; % number of columns
r = 10; % true lower dim rank
W_true = rand([m,r]); % random matrix of size m x r in interval (0,1)
H_true = rand([r,n]); % random matrix of size r x n in interval (0,1)
scale = 10; % relative scale of signal vs random noise
D = scale*W_true*H_true;% + rand([m,n]); % low-rank data + random noise in (0,1)

% create a mask
p = 1.0; % probability of elements kept
M = rand([m,n])<=p; % binary mask of elemeents to use (0 encodes element missing)

[W,H] = masked_nnmf(D, M, r, 'init_mode', 'kmeans');
