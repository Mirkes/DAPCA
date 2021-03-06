load('C:\Datas\DAPCA\MNIST_M\MNIST_M_test.mat');
X1 = data;
y1 = labels;
load('C:\Datas\DAPCA\MNIST_M\MNIST_test.mat');
X2 = data;
y2 = labels;
X = [X1;X2];
labels = [y1';y2'];
figure; [v,u,s] = pca(X,'NumComp',2); scatter(u(:,1),u(:,2),10,labels,'filled'); title('Normal PCA');
figure; alpha = 1; [ V, D ] = SupervisedPCA(X, labels, 2, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));

figure; [v,u,s] = pca(X1,'NumComp',2); scatter(u(:,1),u(:,2),10,y1,'filled'); title('Normal PCA');
figure; alpha = 1; [ V, D ] = SupervisedPCA(X1, y1, 2, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));