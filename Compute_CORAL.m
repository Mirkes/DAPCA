Xs = load('C:\MyPrograms\Python\DAPCA\2clusters_3d_X.csv');
Xt = load('C:\MyPrograms\Python\DAPCA\2clusters_3d_Y.csv');
labels = load('C:\MyPrograms\Python\DAPCA\2clusters_3d_labels.csv');
target_labels = load('C:\MyPrograms\Python\DAPCA\2clusters_3d_target_labels.csv');

%Xs = Xs ./ repmat(sum(Xs,2),1,size(Xs,2));
%Xs = Xs - repmat(mean(Xs,1),size(Xs,1),1);
Xs = zscore(Xs,1);

%Xt = Xt./ repmat(sum(Xt,2),1,size(Xt,2));
%Xt = Xt - repmat(mean(Xt,1),size(Xt,1),1);
Xt = zscore(Xt,1);

eps = 1;
cov_source = cov(Xs) + eps*eye(size(Xs, 2));
cov_target = cov(Xt) + eps*eye(size(Xt, 2));
A_coral = cov_source^(-1/2)*cov_target^(1/2);
Xs_coral = Xs * A_coral;
%Xs_coral = Xs;

ind1 = labels==1;
ind2 = labels==2;
target_ind1 = target_labels==1;
target_ind2 = target_labels==2;


subplot(2,2,1);
i1 = 1;
i2 = 2;
plot(Xs_coral(ind1,i1),Xs_coral(ind1,i2),'g.');
hold on;
plot(Xs_coral(ind2,i1),Xs_coral(ind2,i2),'y.');
plot(Xt(target_ind1,i1),Xt(target_ind1,i2),'r.');
plot(Xt(target_ind2,i1),Xt(target_ind2,i2),'b.');
xlabel(['x' num2str(i1)]);
ylabel(['x' num2str(i2)]);

subplot(2,2,2);
i1 = 1;
i2 = 3;
plot(Xs_coral(ind1,i1),Xs_coral(ind1,i2),'g.');
hold on;
plot(Xs_coral(ind2,i1),Xs_coral(ind2,i2),'y.');
plot(Xt(target_ind1,i1),Xt(target_ind1,i2),'r.');
plot(Xt(target_ind2,i1),Xt(target_ind2,i2),'b.');
xlabel(['x' num2str(i1)]);
ylabel(['x' num2str(i2)]);


subplot(2,2,3);
i1 = 2;
i2 = 3;
plot(Xs_coral(ind1,i1),Xs_coral(ind1,i2),'g.');
hold on;
plot(Xs_coral(ind2,i1),Xs_coral(ind2,i2),'y.');
plot(Xt(target_ind1,i1),Xt(target_ind1,i2),'r.');
plot(Xt(target_ind2,i1),Xt(target_ind2,i2),'b.');
xlabel(['x' num2str(i1)]);
ylabel(['x' num2str(i2)]);

subplot(2,2,4);
XX = [Xs_coral;Xt];
[v,u,s] = pca(XX);
Xsp = u(1:size(Xs,1),1:2);
Xtp = u(size(Xs,1)+1:end,1:2);
plot(Xsp(ind1,1),Xsp(ind1,2),'g.');
hold on;
plot(Xsp(ind2,1),Xsp(ind2,2),'y.');
plot(Xtp(target_ind1,1),Xtp(target_ind1,2),'r.');
plot(Xtp(target_ind2,1),Xtp(target_ind2,2),'b.');
xlabel('PCA1');
ylabel('PCA2');


set(gcf,'Position',[213.0000   15.3333  654.0000  626.0000]);

csvwrite('C:\MyPrograms\Python\DAPCA\2clusters_3d_X_coral.csv',Xs_coral);
