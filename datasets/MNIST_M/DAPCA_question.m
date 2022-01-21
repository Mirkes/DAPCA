dataset = 'ex479';
load(['datasets/MNIST_M/MNIST_',dataset,'.mat']);
X1 = data;
y1 = labels;
disp(['X1 shape',num2str(size(X1))])

[v,u] = pca(X1);
scatter(u(:,1),u(:,2),10,y1,'filled'); title('PCA of X only');
mn = mean(X1);

alpha = 1;
[V1, D1, PX, PY] = DAPCA(X1, y1, [], 2, 'alpha', alpha);
% Drawing
figure; 
scatter(PX(:,1), PX(:,2), 10, y1, 'filled'); 
title(sprintf('SPCA through DAPCA, alpha=%f', alpha));

V(:,1) = mean(X1(y1==4,:))-mean(X1(y1==9,:));
V(:,2) = mean(X1(y1==9,:))-mean(X1(y1==7,:));
% orthogonalization just in case
V(:,2) = V(:,2) - sum(V(:,1).*V(:,2))/sum(V(:,1).*V(:,1))*V(:,1);
Xp = X1*V;
figure;
scatter(Xp(:,1),Xp(:,2),10,y1);
title('Simplest separation of classes')
