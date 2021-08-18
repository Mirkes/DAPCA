load('C:\Datas\DAPCA\Amazon_Reviews\dvd_test.mat');
X = data;
figure; [v,u,s] = pca(X,'NumComp',2); scatter(u(:,1),u(:,2),10,labels,'filled'); title('Normal PCA');
figure; alpha = 1; [ V, D ] = SupervisedPCA(X, labels, 2, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));