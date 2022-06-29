W = ones(size(X,1),size(X,1));
alpha = 1.0
ind = find(labels==4); for i=1:length(ind) for j=1:length(ind) W(ind(i),ind(j))=-alpha; end; end
ind = find(labels==7); for i=1:length(ind) for j=1:length(ind) W(ind(i),ind(j))=-alpha; end; end
ind = find(labels==9); for i=1:length(ind) for j=1:length(ind) W(ind(i),ind(j))=-alpha; end; end
q = sum(W);
Q1_ = zeros(size(X,2),size(X,2));
for l=1:size(Q1_,1) for m=1:size(Q1_,1) for i=1:size(X,1) Q1_(l,m)=Q1_(l,m)+q(i)*X(i,l)*X(i,m); end; end; end;
Q2_ = X'*W*X;
[V_, D_] = eig(Q1_-Q2_); P_=X*V_; plot(diag(D_)); scatter(P_(:,1),P_(:,2),10,labels)

[labs, ~, labNum] = unique(labels);
nClass = length(labs);