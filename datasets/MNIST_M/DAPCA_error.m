load('datasets/MNIST_M/MNIST_test_small.mat');
X1 = data;
y1 = labels;

[v,u] = pca(X1); scatter(u(:,1),u(:,2),10,y1,'filled');

alpha = 1; 
[ V, D ] = SupervisedPCA(X1, y1, 2, alpha); u = X1*V; 
figure; scatter(u(:,1),u(:,2),10,y1,'filled');

[V1, D1, PX, PY] = DAPCA(X1, y1, [], 2, 'alpha', alpha);
figure;  scatter(PX(:,1), PX(:,2), 10, y1, 'filled'); 