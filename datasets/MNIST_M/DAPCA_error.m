dataset = 'test_small_PCA'
load(['datasets/MNIST_M/MNIST_',dataset,'.mat']);
X1 = data;
y1 = labels;
load(['datasets/MNIST_M/MNIST_M_',dataset,'.mat']);
X2 = data;
y2 = labels;
disp(['X1 shape',num2str(size(X1))])
disp(['X2 shape',num2str(size(X2))])

alpha = 5;
beta = 1;
gamma = 0.02;
kNN = 5;
[V2, D2, PXd, PYd] = DAPCA(X1, y1, X2, 2, 'alpha', alpha, 'beta', beta, 'gamma', gamma,'kNN',kNN);
