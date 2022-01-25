load('C:\Datas\DAPCA\MNIST_M\MNIST_ex479.mat');
X = data;
y = labels;
alpha = 1; 
figure; [v,u,s] = pca(X,'NumComp',2); scatter(u(:,1),u(:,2),10,labels,'filled'); title('Normal PCA');
figure; [ V, D ] = SupervisedPCA(X, labels, 2, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));
figure; [V1, D1, PX, PY] = DAPCA(X, labels, [], 2, 'alpha', alpha); scatter(PX(:,1), PX(:,2), 10, labels, 'filled'); title(sprintf('SPCA through DAPCA, alpha=%f', alpha));
