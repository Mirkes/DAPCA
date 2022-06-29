d1 = load('C:/Datas/DAPCA/MNIST_M/MNIST_train_codedNN.mat');
d2 = load('C:/Datas/DAPCA/MNIST_M/MNIST_M_train_codedNN.mat');
X1 = double(d1.data);
X2 = double(d2.data);
[v,u,s] = pca([X1;X2],'NumComponents',2);
scatter(u(1:size(X1,1),1),u(1:size(X1,1),2),2,'b.'); hold on; scatter(u(size(X1,1)+1:size(u,1),1),u(size(X1,1)+1:size(u,1),2),2,'r.');

dSA1 = struct(); 
dSA2 = struct();

for i=5:5:100

addpath('./utils');
[X1p,X2p] = subspace_alignment(X1,X2,i);
[v,up,s] = pca([X1p;X2p],'NumComponents',2);
figure;
scatter(up(1:size(X1,1),1),up(1:size(X1,1),2),2,'b.'); hold on; scatter(up(size(X1,1)+1:size(up,1),1),up(size(X1,1)+1:size(up,1),2),2,'r.');
title(num2str(i))
dSA1.(['data',num2str(i)]) = X1p;
dSA2.(['data',num2str(i)]) = X2p;
end

dSA1.labels = d1.labels;
dSA2.labels = d2.labels;
save('c:/Datas/DAPCA/MNIST_M/MNIST_train_coded_SA.mat','-struct','dSA1');
save('c:/Datas/DAPCA/MNIST_M/MNIST_M_train_coded_SA.mat','-struct','dSA2');
