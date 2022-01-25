X = load('C:\Datas\DAPCA\2clusters\2clusters_3d\X.csv');
labels = load('C:\Datas\DAPCA\2clusters\2clusters_3d\labels.csv');
Y = load('C:\Datas\DAPCA\2clusters\2clusters_3d\Y.csv');
labelsY = -ones(2000,1);
labels_all = [labels;labelsY];
XX = [X;Y];

alpha = 10; 
gamma = 0.0015;
maxIter = 30;
beta = 1;
kNN = 1;

figure; [v,u,s] = pca(X); scatter(u(:,1),u(:,2),10,labels,'filled'); title('Normal PCA');
figure; 
histogram(u(labels==1),30,'FaceColor','b','FaceAlpha',.3); hold on; 
histogram(u(labels==2),30,'FaceColor','y','FaceAlpha',.3); 
title('Normal PCA');

figure; [ V, D ] = SupervisedPCA(X, labels, 2, alpha); u = X*V; scatter(u(:,1),u(:,2),10,labels,'filled'); title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));
figure; 
histogram(u(labels==1),30,'FaceColor','b','FaceAlpha',.3); hold on; 
histogram(u(labels==2),30,'FaceColor','y','FaceAlpha',.3);
title(strcat('Supervised advanced PCA, alpha=',num2str(alpha)));

figure; u = XX*V; scatter(u(:,1),u(:,2),10,labels_all,'filled'); title(strcat('Supervised advanced PCA, all points, alpha=',num2str(alpha)));
figure;
histogram(u(labels_all==1,1),30,'FaceColor','g','FaceAlpha',.3); hold on; 
histogram(u(labels_all==2,1),30,'FaceColor','y','FaceAlpha',.3);
histogram(u(labels_all==-1,1),30,'FaceColor','b','FaceAlpha',.3);
title(strcat('Supervised advanced PCA, all points, alpha=',num2str(alpha)));

figure; 
[V2, D2, PXd, PYd] = DAPCA(X, labels, Y, 2, 'alpha', alpha, 'beta', beta, 'gamma', gamma,'kNN',kNN);
scatter(PXd(:,1),PXd(:,2),10,labels,'filled'); hold on;
scatter(PYd(:,1),PYd(:,2),10,'r','filled'); 
title(strcat('DAPCA, alpha=',num2str(alpha)));
figure;
histogram(PXd(labels_all==1,1),30,'FaceColor','g','FaceAlpha',.3); hold on; 
histogram(PXd(labels_all==2,1),30,'FaceColor','y','FaceAlpha',.3);
histogram(PYd(:,1),30,'FaceColor','b','FaceAlpha',.3);
title(strcat('DAPCA, alpha=',num2str(alpha)));


figure; 
[V2, D2, PXd, PYd] = DAPCA_old(X, labels, Y, 2, 'alpha', alpha, 'beta', beta, 'gamma', gamma,'kNN',kNN);
scatter(PXd(:,1),PXd(:,2),10,labels,'filled'); hold on;
scatter(PYd(:,1),PYd(:,2),10,'r','filled'); 
title(strcat('DAPCA old, alpha=',num2str(alpha)));
figure;
histogram(PXd(labels_all==1,1),30,'FaceColor','g','FaceAlpha',.3); hold on; 
histogram(PXd(labels_all==2,1),30,'FaceColor','y','FaceAlpha',.3);
histogram(PYd(:,1),30,'FaceColor','b','FaceAlpha',.3);
title(strcat('DAPCA old, alpha=',num2str(alpha)));
