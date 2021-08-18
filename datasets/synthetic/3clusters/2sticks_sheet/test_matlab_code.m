X = load('C:\Datas\DAPCA\3clusters\2sticks_sheet\X.csv');
labels = load('C:\Datas\DAPCA\3clusters\2sticks_sheet\labels.csv');
figure; [v,u,s] = pca(X); scatter(u(:,1),u(:,2),10,labels,'filled'); title('Normal PCA');
figure; [ V, D ] = SupervisedPCA(X, labels, 2, 'usual'); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title('Usual PCA');
figure; [ V, D ] = SupervisedPCA(X, labels, 2, 'super'); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title('Supervised PCA');
figure; [ V, D ] = SupervisedPCA(X, labels, 2, 'supernorm'); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title('Supervised normalized PCA');
figure; alpha = 1; [ V, D ] = SupervisedPCA(X, labels, 2, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));