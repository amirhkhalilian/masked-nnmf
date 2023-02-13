clc; clear; close all

% load data
D = load('./tests/Hubble.mat', 'M').M; % data is stored in veriable M in the mat file
% create a random mask -- keep p% of enteries
p = 0.6;
M = rand(size(D))<=p;
% data with missing enteries
MD = M.*D;
% set a rank
r = 8;
% run the solver
[W,H] = masked_nnmf(D, M, r,...
                    'init_mode', 'rand',...
                    'maxiter', 250);
% recovered estimated data 
D_hat = W*H;
abs_Err = abs(D-D_hat);

figure;
tiledlayout(4,10, 'Padding', 'none', 'TileSpacing', 'compact');
for i = 1:10
    nexttile
    plot_columns(D(:,i),128,128);
end
for i = 1:10
    nexttile
    plot_columns(MD(:,i),128,128);
end
for i = 1:10
    nexttile
    plot_columns(D_hat(:,i),128,128);
end
for i = 1:10
    nexttile
    plot_columns(abs_Err(:,i),128,128);
end


function plot_columns(X, x, y)
    imagesc(reshape(X, x, y));
    colormap(flipud(gray(256)));
    axis image; axis off;
end

