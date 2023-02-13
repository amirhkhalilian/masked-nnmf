function [W, H, opts] = masked_nnmf(D, M, r, varargin)
    % Solver for masked non-negative matrix factorization problem
    % given D matrix of size m x n 
    % and binary mask matrix of size m x n (entry ij =0 if D_{ij} missing)
    % the solver finds W and H by solving
    % argmin |M .* (D - WH)|_F^2 s.t. W>=0 size m x r and H>=0 size r x n
    % Inputs:
    % D: data matrix of size m x n
    % M: mask matrix of size m x n 0 entries encode missing values in D
    % r: number of columns and rows in W and H, respectively.
    % Name, Value: pairs of optimization parameters
    % 'maxiter': maximum number of iterations to run
    % 'verbose': flag to print iteration information
    % 'use_parallel': flag to use matlab parallel processing
    % 'num_cores': number of cores to use if use_parallel is set
    % 'tol': tolerance to halt the iteration
    % 'init_mode': initialization mode
    % Outputs:
    % W: matrix of size m x r
    % H: matrix of size r x n
    % opts: optimization results

    % parse the varargin
    p = inputParser;
    addParameter(p, 'maxiter', 100);
    addParameter(p, 'verbose', true);
    addParameter(p, 'use_parallel', false);
    addParameter(p, 'num_cores', 10);
    addParameter(p, 'tol', 1e-4);
    addParameter(p, 'init_mode', 'rand');
    parse(p, varargin{:});
    maxiter = p.Results.maxiter;
    verbose = p.Results.verbose;
    use_parallel = p.Results.use_parallel;
    num_cores = p.Results.num_cores;
    tol = p.Results.tol;
    init_mode = p.Results.init_mode;

    % check if parallel processing is available if requested
    if use_parallel
        p = gcp('nocreate');
        if isempty(p)
            % init parpool
            flag_init_parpool = true;
        else
            % parpool exists... use what exists
            num_cores = p.NumWorkers;
            flag_init_parpool = false;
        end
    else
        num_cores = 0;
    end

    % pre process the mask to get valid rows and cols
    [rows, cols] = ind2sub(size(M), find(M));
    cols2use = cell(1,size(D,1));
    rows2use = cell(1,size(D,2));
    parfor (i=1:size(D,1), num_cores) % loop over rows and save cols 2 use
        cols2use{i} = cols(find(rows==i));
    end
    parfor (j=1:size(D,2), num_cores) % loop over cols and save rows 2 use
        rows2use{j} = rows(find(cols==j));
    end
    clearvars rows cols

    % optimization records
    opts.loss = zeros([maxiter, 1]);
    opts.max_abs_err = zeros([maxiter, 1]);

    % initialize W
    if strcmp(init_mode, 'rand')
        W = rand([size(D,1), r]); % random non-negative init
        H = zeros([r, size(D,2)]);
    elseif strcmp(init_mode, 'kmeans')
        [~, W] = kmeans((D.*M)', r,...
                        'Distance', 'cosine',...
                        'Replicates', 5,...
                        'maxiter', 500);
        W = transpose(W);
        H = zeros([r, size(D,2)]);
    elseif strcmp(init_mode, 'nnmf')
        [W, H] = nnmf((D.*M), r,...
                      'Replicates', 5);
    else
        error(['init_mode:' , init_mode, 'is not valid!']);
    end
    for itr = 1:maxiter
        % update H
        WT = W';
        parfor (j = 1:size(D,2), num_cores) % loop over columns of D
            WT_mj_dj = transpose(sum(W(rows2use{j},:).*D(rows2use{j},j),1));
            WT_mj_W = WT(:,rows2use{j}) * W(rows2use{j},:);
            H(:,j) = nnlsm_blockpivot(WT_mj_W, WT_mj_dj, true);
        end
        % update W
        HT = H';
        parfor (i = 1:size(D,1), num_cores) % loop over rows of D
            H_mi_di = sum(H(:,cols2use{i}).*D(i,cols2use{i}),2);
            H_mi_H = H(:,cols2use{i}) * HT(cols2use{i},:);
            W(i,:) = nnlsm_blockpivot(H_mi_H, H_mi_di, true);
        end
        % log the loss function
        res = M.*(D-W*H);
        opts.loss(itr) = norm(res, 'fro');
        opts.loss_abs(itr) = max(abs(res),[],'all');
        if verbose
            fprintf('itr: %3d, loss:%2.2e, abs_err:%2.2e\n',...
                     itr, opts.loss(itr), opts.loss_abs(itr));
        end
        % check stopping criteria
        if itr>1 && opts.loss(itr-1)-opts.loss(itr)<tol
            opts.tolerance_reached = true;
            break
        end
    end
end
