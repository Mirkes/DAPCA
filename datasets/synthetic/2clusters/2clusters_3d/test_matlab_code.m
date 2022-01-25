X = load('C:\Datas\DAPCA\2clusters\2clusters_3d\X.csv');
labels = load('C:\Datas\DAPCA\2clusters\2clusters_3d\labels.csv');
Y = load('C:\Datas\DAPCA\2clusters\2clusters_3d\Y.csv');
labelsY = -ones(1500,1);
labels_all = [labels;labelsY];
XX = [X;Y]
figure; [v,u,s] = pca(X); scatter(u(:,1),u(:,2),10,labels,'filled'); title('Normal PCA');
figure; [ V, D ] = SupervisedPCA(X, labels, 2, 'usual'); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title('Usual PCA');
figure; [ V, D ] = SupervisedPCA(X, labels, 2, 'super'); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title('Supervised PCA');
figure; [ V, D ] = SupervisedPCA(X, labels, 2, 'supernorm'); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title('Supervised normalized PCA');
figure; alpha = 1; [ V, D ] = SupervisedPCA(X, labels, 2, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));
figure; alpha = 1; [ V, D ] = SupervisedPCA(XX, labels_all, 2, alpha); u = XX*V; scatter(u(:,1),u(:,2),10,labels_all,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));
