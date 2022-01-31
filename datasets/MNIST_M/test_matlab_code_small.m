close all;
load('C:\Datas\DAPCA\MNIST_M\MNIST_ex479.mat');
X = data;
%ind = find((labels==4)|(labels==7));
%X = X(ind,:)
%labels = labels(ind)
y = labels;
%X = load('C:\Datas\DAPCA\2clusters\2clusters_3d\X.csv');
%labels = load('C:\Datas\DAPCA\2clusters\2clusters_3d\labels.csv');
alpha = 5.0;
ncomp = 169
figure; [v,u,s] = pca(X,'NumComp',ncomp); scatter(u(:,1),u(:,2),10,labels,'filled'); title('Normal PCA');
figure; [ V, D, L ] = SupervisedPCA_exp(X, labels, ncomp, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));
%figure; [V1, D1, PX, PY, Q1, Q2, delta] = DAPCA_exp(X, labels, [], ncomp, 'alpha', alpha); PX = X*V1; scatter(PX(:,1), PX(:,2), 10, labels, 'filled'); title(sprintf('SPCA through DAPCA, alpha=%f', alpha));
figure; [V1, D1, PX, PY] = DAPCA(X, labels, [], ncomp, 'alpha', alpha); PX = X*V1; scatter(PX(:,1), PX(:,2), 10, labels, 'filled'); title(sprintf('SPCA through DAPCA, alpha=%f', alpha));
figure; [V1, D1, PX, PY] = DAPCA_old(X, labels, [], ncomp, 'alpha', alpha); PX = X*V1; scatter(PX(:,1), PX(:,2), 10, labels, 'filled'); title(sprintf('SPCA through DAPCA old, alpha=%f', alpha));
