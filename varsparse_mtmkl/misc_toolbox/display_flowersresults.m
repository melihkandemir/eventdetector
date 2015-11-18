display('Using M = 20')
load flowers_results20

Q = 102;
auc = zeros(1,Q);
bep = zeros(1,Q);
for q = 1:Q
    [ P, OP, A, T ] = roc_caltech([outTs(:,q) T_tst(:,q)>0]);
    auc(q) = A;
    bep(q) = OP;
end

AUC=mean(auc)
BEP=mean(bep)

[val, Testclass]=max(T_tst,[],2);
[val, Predclass]=max(outTs,[],2);
CorrectClassRate = mean(Testclass==Predclass)


display('Using M = 104')
load flowers_results100

Q = 102;
auc = zeros(1,Q);
bep = zeros(1,Q);
for q = 1:Q
    [ P, OP, A, T ] = roc_caltech([outTs(:,q) T_tst(:,q)>0]);
    auc(q) = A;
    bep(q) = OP;
end

AUC=mean(auc)
BEP=mean(bep)

[val, Testclass]=max(T_tst,[],2);
[val, Predclass]=max(outTs,[],2);
CorrectClassRate = mean(Testclass==Predclass)


display('Using 102 standard GPs with covariance the sum of the 4 cov types')
load standardgp_results

Q = 102;
auc = zeros(1,Q);
bep = zeros(1,Q);
for q = 1:Q
    [ P, OP, A, T ] = roc_caltech([ps(:,q) T_tst(:,q)>0]);
    auc(q) = A;
    bep(q) = OP;
end

AUC=mean(auc)
BEP=mean(bep)

[val, Testclass]=max(T_tst,[],2);
[val, Predclass]=max(ps,[],2);
CorrectClassRate = mean(Testclass==Predclass)
