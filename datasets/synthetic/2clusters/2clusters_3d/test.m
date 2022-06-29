clear all;

alpha = 10;
gamma = 0.015;
beta = 1;
delta = 1;
kNN = 1;
nComp = 2;

subsampleX = 100;
subsampleY = 100;

close all;


X = load('C:\Datas\DAPCA\2clusters\2clusters_3d\X.csv');
labels = load('C:\Datas\DAPCA\2clusters\2clusters_3d\labels.csv');
inds = [[1:subsampleX],[1001:1000+subsampleX]];
X = X(inds,:);
labels = labels(inds);
Y = load('C:\Datas\DAPCA\2clusters\2clusters_3d\Y.csv');
labelsY = -ones(2000,1);
inds = [[1:subsampleY],[1001:1000+subsampleY]];
Y = Y(inds,:);
labelsY = labelsY(inds);
labels_all = [labels;labelsY];
Xa = [X;Y];

[nX,d] = size(X);
nY = size(Y,1);

%Xa = [X',Y']';
subplot(2,3,1); [v,u,s] = pca(Xa); 
scatter(u(1:nX,1),u(1:nX,2),5,labels,'filled'); hold on; title('Normal PCA');
scatter(u(nX+1:nX+nY,1),u(nX+1:nX+nY,2),1,'g','filled'); 
axis('equal');

[V,D] = SupervisedPCA(X, labels, 2, alpha); PX = X*V; PY=Y*V; 
subplot(2,3,2); scatter(PX(:,1),PX(:,2),10,labels,'filled'); hold on; 
scatter(PY(:,1),PY(:,2),1,'g','filled'); 
title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));
axis('equal');

% ----------------- COMPUTING DAPCA --------------------

%[V2, D2, PXd, PYd, wXXd, wYYd] = DAPCA_old_exp(X, labels, Y, 2, 'alpha', alpha, 'beta', beta, 'gamma', gamma,'kNN',kNN,'maxIter',1);
[V2, D2, PXd, PYd] = DAPCA(X, labels, Y, nComp, 'alpha', alpha, 'beta', beta, 'gamma', gamma,'kNN',kNN,'maxIter',1);
subplot(2,3,4); scatter(PXd(:,1),PXd(:,2),5,labels,'filled'); hold on;
scatter(PYd(:,1),PYd(:,2),1,'g','filled'); 
axis('equal')
title(['DAPCA, alpha=',num2str(alpha),',gamma=',num2str(gamma)]);


% ---------------------- Replication code

beta = beta / (nY * (nY - 1));
gamma = gamma / (kNN * nY);


[labs, ~, labNum] = unique(labels);
nClass = length(labs);
cnt = zeros(nClass, 1);
means = zeros(nClass, d);
for k = 1:nClass
    ind = labNum == k;
    cnt(k) = sum(ind);
    means(k, :) = sum(X(ind, :));
end
meanY = sum(Y);

delta = ones(nClass);
alpha_old = alpha
alpha = -alpha ./ (cnt .* (cnt -1));
delta = delta ./ (cnt * cnt');
tmp = triu(delta, 1);
delta = tmp + tmp' + diag(alpha);

disp('delta')
disp(delta)

constQ = beta * (meanY' * meanY);
    for k = 1:nClass
        % diagonal part
        constQ = constQ + delta(k, k) * (means(k, :)' * means(k, :));
        % Off diagonal part
        for kk = k + 1:nClass
            tmp = delta(k, kk) * (means(k, :)' * means(kk, :));
            constQ = constQ + tmp + tmp';
        end
    end    
Q2 = constQ;

PX = X; PY = Y; PX2 = sum(PX .^ 2, 2)'; 
wY = repmat(nY * beta, nY, 1);
tmp = delta * cnt;
wX = repelem(tmp, cnt);

%------------------- begin to compute the modified matrix --------------------

wXXi = wX;
wYYi = wY;

dist = sum(PY.^2, 2) + PX2 - 2 * PY * PX';
[dist, ind] = sort(dist, 2);
kNNDist = - gamma * ones(size(dist(:, 1:kNN)));
kNNs = ind(:, 1:kNN);

wYYi = wYYi + sum(kNNDist, 2);
tmp = zeros(nY, nX);
tmp(sub2ind(size(tmp),repmat((1:nY)', 1, kNN), ind(:, 1:kNN))) = kNNDist;
tmpi = tmp;
tmp = Y' * tmp * X;
Q2 = Q2 + tmp + tmp';

wXXi = wXXi + sum(kNNDist, 1);
%wXXi = wXXi + accumarray(kNNs(:), kNNDist(:), [nX, 1]);
Q1 = X' * (wXXi .* X) + Y' * (wYYi .* Y);

Q = Q1-Q2;

[Vi_,Di_] = eig(Q); PX = X*Vi_; PY = Y*Vi_; 

        Di_ = diag(Di_); 
        % Sort eigenvalues
        [Di_, ind] = sort(Di_, 'descend');
        Vi_ = Vi_(:, ind);
        PX = X * Vi_;
	PY = Y*Vi_;
        Di_ = Di_(1:nComp);
        Vi_ = Vi_(:, 1:nComp);
        % Standardise direction
        ind = sum(Vi_) < 0;
        Vi_(:, ind) = - Vi_(:, ind);


subplot(2,3,3); scatter(PX(:,1),PX(:,2),5,labels); hold on; 
scatter(PY(:,1),PY(:,2),1,'g'); title('After implicit DAPCA iteration');
axis('equal');


%--------------------- Computing DAPCA iteration explicitly

[V,D] = SupervisedPCA(X, labels, 2, alpha_old); PX = X*V; PY=Y*V; 
subplot(2,3,5); scatter(PX(:,1),PX(:,2),5,labels); hold on; 
scatter(PY(:,1),PY(:,2),1,'g'); title('Before explicit DAPCA iteration');
axis('equal');

PX = X;
PY = Y;
PX2 = sum(PX .^ 2, 2)'; 

for i=1:1

disp(['Iteration ',num2str(i)])

W = zeros(nX+nY);
W(1:cnt(1),1:cnt(1)) = alpha(1);
W(cnt(1)+1:nX,1:cnt(1)) = delta(1,2);
W(1:cnt(1),cnt(1)+1:nX) = delta(2,1);
W(cnt(1)+1:nX,cnt(1)+1:nX) = alpha(2);
W(nX+1:nX+nY,nX+1:nX+nY) = beta;
tmp = zeros(nY, nX);
dist = sum(PY.^2, 2) + PX2 - 2 * PY * PX';
[dist, ind] = sort(dist, 2);
kNNDist = - gamma * ones(size(dist(:, 1:kNN)));
tmp(sub2ind(size(tmp),repmat((1:nY)', 1, kNN), ind(:, 1:kNN))) = kNNDist;
tmpe = tmp;
W(1:nX,nX+1:nX+nY) = tmp';
W(nX+1:nX+nY,1:nX) = tmp;

wXX = sum(W(1:nX,:),2);
wYY = sum(W(nX+1:nX+nY,:),2);
Q1_ = X' * (wXX .* X) + Y' * (wYY .* Y);
Q2_ = Xa'*W*Xa;
[Ve_,De_] = eig(Q1_-Q2_); PX = X*Ve_; PY = Y*Ve_; 

        De_ = diag(De_); 
        % Sort eigenvalues
        [De_, ind] = sort(De_, 'descend');
        Ve_ = Ve_(:, ind);
        PX = X * Ve_;
	PY = Y*Ve_;
        De_ = De_(1:nComp);
        Ve_ = Ve_(:, 1:nComp);
        % Standardise direction
        ind = sum(Ve_) < 0;
        Ve_(:, ind) = - Ve_(:, ind);


end

subplot(2,3,6); scatter(PX(:,1),PX(:,2),5,labels); hold on; 
scatter(PY(:,1),PY(:,2),1,'g'); title('After explicit DAPCA iteration')
axis('equal');

set(gcf,'Position',[181,51,1006,545])

disp('D_imlicit')
disp(Di_)
disp('D_explicit')
disp(De_)
disp('D_DAPCA')
disp(D2)
