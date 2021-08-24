function [V, D, PX, PY] = DAPCA(X, labels, Y, nComp, varargin)
% DAPCA calculated Domain Adaptation Principal Components (DAPCA) or,
% if matrix Y is empty, Supervised principal components (SPCA) for problem
% of classification. Detailed description of these types of principal
% components (PCom) can be found in 
%   Gorban, A.N., Grechuk, B., Mirkes, E.M., Stasenko, S.V. and Tyukin,
%   I.Y., 2021. High-dimensional separability for one-and few-shot
%   learning. arXiv preprint arXiv:2106.15416. or in ???
%
% Usage
%           [V, D, PX, PY] = DAPCA(X, Labels, Y, nComp)
% or
%           [V, D, PX, PY] = DAPCA(X, Labels, Y, nComp, Name, Value)
%
% Inputs:
%   X is n-by-d labelled data matrix. Each row of matrix contains one
%       observation (case, record, object, instance).
%   labels is n-by-1 vector of labels. Each unique value is considered as
%       one class. Number of classes nClass is equal to number of unique
%       values in array labels.
%   Y is m-by-d unlabelled data matrix. Each row of matrix contains one
%       observation (case, record, object, instance).
%   nComp is number of required PComs and must be positive integer number
%       less than d.
%   Name, Value pairs can be one of the following type
%       'alpha', nonnegative real number is attraction of projections of
%           points with the same label in labelled dataset. 
%           Default value is 0.
%       'alpha', nClass-by-1 vector of nonnegative real numbers is vector
%           of attraction of projections of points with the same label in
%           labelled dataset. In this case each class have individual
%           coefficient of attraction.
%       'beta', positive real number is repulsion of projections of points
%           of unlabelled dataset.
%           Default value is 0.9.
%       'gamma', positive real number is attraction between projection of
%           unlabelled point and k nearest labelled neighbours.
%           Default value is 0.4.
%       'kNN', positive integer number is number of labelled nearest
%           neighbours to use for attraction of each unlabbeled data point.
%           Default value 1.
%       'kNNweights', 'Uniform' required usage of the same weghts for all k
%           neares neighbours.
%           It is default value
%       'kNNweights', 'dist' assumed usage of weights proportional to
%           distances. This function is implemented as
%               function weights = distProp(distances)
%                   weights = distances ./ sum(distnaces, 2);
%               end
%       'kNNweights', function handle is handle of function inputs and
%           outputs described above.
%       'delta', positive real number assumes usage the same number for all
%           pairs of classes.
%           Default value is 1.
%       'delta', nClass-by-1 vector of positive real numbers R assumes
%           usage of matrix Delta with elements Delta(i,j) = abs(R(i)-R(j))
%           for i < j. 
%       'delta', matrix with positive real numbers upper the main diagonal.
%           Matrix must have size nClass-by-nClass. This method can be
%           usefull for ordinal target attribute.
%       'maxIter', positive integer number is maximal number of itterations
%           for iterative DAPCA.
%           Default value is 5.
%
% Outputs:
%   V is d-by-nComp matrix with one PCom is each column.
%   D is nComp-by-1 vector with the greatest nComp normalised eigenvalues:
%       eigenvalues divided by sum of all eigenvalues.
%   PX is n-by-nComp matrix of projections of X onto nComp PComs.
%   PY is m-by-nComp matrix of projections of Y onto nComp PComs.
%
%   Data matrices X and Y MUST be complete: usage of NaN is forbidden. To
%   impute missing values you can use any appropriate methods. For example
%   you can use kNNImpute or svdWithGaps from the Github repository
%   https://github.com/Mirkes/DataImputation.
%   Rows with NaN valuse will be removed.

    % Sanity check of positional arguments
    % Type of X
    if ~isnumeric(X) || ~ismatrix(X)
        error('the first argument must be numerical matrix');
    end
    % Type of labels is arbitrary. Nothing to check. Convert to column
    % vector.
    labels = labels(:);
    % Remove NaNs from X
    ind = sum(isnan(X), 2);
    ind = ind > 0;
    if sum(ind) > 0 
        warning(['Matrix X contains %d rows with missing values.',...
            'All these rows were deleted'],sum(ind));
        X(ind, :) = [];
        labels(ind) = [];
    end
    % Calculate sizes of data
    [nX, d] = size(X);
    if nX == 0
        error(['Matrix X must not be empty even after removing of',...
            ' incomplete rows']);
    end
    k = size(labels, 1);
    if k ~= nX 
        error(['Number of elements in labels must be the same as number',...
            ' of row in matrix X']);
    end
    [labs, ~, labNum] = unique(labels);
    nClass = length(labs);
    % Type of matrix Y
    if ~isempty(Y)
        ind = sum(isnan(Y), 2);
        ind = ind > 0;
        if sum(ind) > 0
            Y(ind, :) = [];
        end
    end
    useY = true;
    if isempty(Y)
        warning('Since matrix Y is empty the SPCA is used');
        useY = false;
    else
        [nY, k] = size(Y);
        if k ~= d
            error('It is assumed that set of features in X and Y is the same');
        end
    end
    % Value of nComp
    if ~isnumeric(nComp) || floor(nComp) ~= nComp || nComp < 1
        error('nComp must be positive integer value');
    end

    % Extraction of arguments from varargin
    % default values
    alpha = zeros(nClass, 1);
    beta = 0.9;
    gamma = 0.4;
    kNN = 1;
    kNNweights = @uniformWeights;
    delta = ones(nClass);
    maxIter = 5;
    % Loop of arguments
    for k = 1:2:length(varargin)
        tmp = varargin{k};
        if ~isstring(tmp) && ~ischar(tmp)
            error('Wrong argument name "%s" in name value pair %d', tmp, k);
        end
        if strcmpi('alpha', tmp)
            alpha = varargin{k + 1};
        elseif strcmpi('beta', tmp)
            beta = varargin{k + 1};
        elseif strcmpi('gamma', tmp)
            gamma = varargin{k + 1};
        elseif strcmpi('kNN', tmp)
            kNN = varargin{k + 1};
        elseif strcmpi('kNNweights', tmp)
            kNNweights = varargin{k + 1};
        elseif strcmpi('delta', tmp)
            delta = varargin{k + 1};
        elseif strcmpi('maxIter', tmp)
            maxIter = varargin{k + 1};
        else
            error('Wrong argument name "%s" in name value pair %d', tmp, k);
        end
    end
    % Analise values of optional arguments
    if isscalar(alpha)
        alpha = alpha * ones(nClass, 1);
    else
        if length(alpha) ~= nClass
            error(['Value of parameter "alpha" must be either',...
                ' nonnegative scalar or vector with number of classes',...
                ' %d nonnegative elements'], nClass);
        end
    end
    if ~isnumeric(beta) || ~isscalar(beta) || beta < 0
        error('Value of parameter "beta" must be positive scalar');
    end
    if ~isnumeric(gamma) || ~isscalar(gamma) || gamma < 0
        error('Value of parameter "beta" must be positive scalar');
    end
    if ~isnumeric(kNN) || floor(kNN) ~= kNN || kNN < 1
        error(['kNN is number of labelled nearest neighbours and must',...
            ' be positive integer value']);
    end
    if isstring(kNNweights) || ischar(kNNweights)
        if strcmpi('Uniform', kNNweights)
            kNNweights = @uniformWeights;
        elseif strcmpi('dist', kNNweights)
            kNNweights = @distProp;
        else
            error('Wrong value of argument "kNNweights"');
        end
    else
        %check of function handle
        if ~isa(kNNweights, 'function_handle')
            error(['Argument "kNNweights" must be function handle with',...
                ' structure "function weights = function_name(distances)",',...
                ' or string constant "Uniform" or string constant "dist".']);
        end
    end
    % do we have error?
    ind = false;
    if ~isnumeric(delta)
        ind = true;
    else
        if isscalar(delta) && delta > 0
            delta = delta * ones(nClass);
        elseif isvector(delta) && all(delta > 0)
            % Form matrix
            tmp = zeros(nClass);
            for k = 1:nClass
                for kk = k+1:nClass
                    tmp(k, kk) = abs(delta(k) - delta(kk));
                end
            end
            delta = tmp;
        elseif ismatrix(delta) && size(delta, 1) == nClass...
                && size(delta, 2) == nClass
            % now we need to check positivity of upper diagonal matrix
            tmp = sum(triu(delta, 1) > 0, 'all');
            if tmp ~= nClass * (nClass - 1) / 2
                ind = true;
            end
        else
            ind = true;
        end
    end
    if ind
        error(['Argument delta must be positive real number, ',...
            'vector with nClass positive elements, or matrix of size ',...
            'nClass-by-nClass with positive real values upper the main diagonal']);
    end
    if ~isnumeric(maxIter) || floor(maxIter) ~= maxIter || maxIter < 1
        error(['maxIter is maximal number of itterations for iterative',...
            ' DAPCA and must  be positive integer value']);
    end

    % Calculate number of cases of each class n_i and means for classes
    % mu_i formula (6)?
    cnt = zeros(nClass, 1);
    means = zeros(nClass, d);
    for k = 1:nClass
        ind = labNum == k;
        cnt(k) = sum(ind);
        means(k, :) = sum(X(ind, :));
    end
    if useY
        meanY = sum(Y);
    end
    % Convert matrix delta to full matrix with -alpha on diagonal and delta
    % off diagonal and with normalisation by number of elements in each
    % class. Formulae (12)
    alpha = -alpha ./ (cnt .* (cnt -1));
    delta = delta ./ (cnt * cnt');
    tmp = triu(delta, 1);
    delta = tmp + tmp' + diag(alpha);
    % Normalise other coefficients
    if useY
        beta = beta / (nY * (nY - 1));
        gamma = gamma / (kNN * nY);
    end
    
    % Calculation of constant parts of sum of weights vectors. Formulae
    % (13) and (10)
    tmp = delta * cnt;
    wX = repelem(tmp, cnt);
    if useY
        wY = repmat(nY * beta, nY, 1);
    end
    
    % Calculation of constant parts of matrix Q.
    % Y part
    if useY
        constQ = beta * (meanY' * meanY);
    else
        constQ = zeros(d);
    end
    % X Part
    for k = 1:nClass
        % diagonal part
        constQ = constQ + delta(k, k) * (means(k, :)' * means(k, :));
        % Off diagonal part
        for kk = k + 1:nClass
            tmp = delta(k, kk) * (means(k, :)' * means(kk, :));
            constQ = constQ + tmp + tmp';
        end
    end    
    
    % Now we are ready for iterations.
    if useY
        kNNs = zeros(nY, kNN);
        kNNDist = kNNs;
        % estimate step of Y records to calculate distances to all X records
        maxY = floor(1e8 / nX);
        if maxY > nY
            maxY = nY;
        end
        maxY = maxY - 1;
        PY = Y;
    end
    PX = X;
    % Start iterations
    iterNum = 0;
    while true
        wXX = wX;
        Q2 = constQ;
        if useY
            % Remember old kNNs
            oldkNN = kNNs;
            % Calculate new kNNs
            % calculate squared length of X vectors
            PX2 = sum(PX .^ 2, 2)';
            k = 1;
            wYY = wY;
            while k <= nY
                % Define end of fragment
                kk = k + maxY;
                if kk > nY
                    kk = nY;
                end
                % Calculate distances
                dist = sum(PY(k:kk, :) .^ 2, 2) + PX2 - 2 * PY(k:kk, :) * PX';
                % Search NN
                [dist, ind] = sort(dist, 2);
                % Get kNN element
                kNNDist(k:kk, :) = - gamma * kNNweights(dist(:, 2:kNN + 1));
                kNNs(k:kk, :) = ind(:, 2:kNN + 1);
                % Correct wYY
                wYY(k:kk) = wYY(k:kk) + sum(kNNDist(k:kk, :), 2);
                % Add summand to Q2
                nS = kk - k + 1;
                tmp = zeros(nS, nX);
                tmp(sub2ind(size(tmp),repmat((1:nS)', 1, kNN), ind(:, 2:kNN + 1))) = kNNDist(k:kk, :);
                tmp = Y(k:kk, :)' * tmp * X;
                Q2 = Q2 + tmp + tmp';
                % Shift k in Y
                k = kk + 1;
            end
            wXX = wXX + sum(kNNDist, 1);
            if all(oldkNN == kNNs)
                break;
            end
        end    
        % X part of Q1
        Q1 = X' * (wXX .* X);
        % Y part of Q1
        if useY
            Q1 = Q1 + Y' * (wYY .* Y);
        end
        % Full amtrix is
        Q = Q1 - Q2;
        % Calculate principal components.
        [V, D] = eigs(Q, nComp, 'largestreal');
        % Normalise eigenvalues
        D = diag(D) / sum(diag(Q));
        % Sort eigenvalues
        [D, ind] = sort(D, 'descend');
        V = V(:, ind);
        % Standardise direction
        ind = sum(V) < 0;
        V(ind, :) = - V(ind, :);
        % Calculate projection
        PX = X * V;
        if useY
            PY = Y * V;
        else
            PY = [];
        end
        iterNum = iterNum + 1;
        if iterNum == maxIter || ~useY
            break;
        end
    end
    fprintf('Nuber of iterations %d\n', iterNum);
end

function weights = uniformWeights(distances)
% Calculate constant weights for all neightbours.
    weights = ones(size(distances));
end

function weights = distProp(distances)
% Calculate weights proportional to relative distance.
    weights = distances ./ max(distnaces, [], 2);
end
