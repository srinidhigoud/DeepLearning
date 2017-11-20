% clear all;
% close all;
% clc;

fileID = fopen('/Users/srinidhigoud/Desktop/assignment1/world.txt','r');
formatSpec = '%lf';
sizeA = Inf;
C = fscanf(fileID,formatSpec,sizeA);
buf=ones(10,1,'double');
X=[C; buf];%Homogeneous form
X=reshape(X,10,4);
fclose(fileID);
fileID = fopen('/Users/srinidhigoud/Desktop/assignment1/image.txt','r');
formatSpec = '%lf';
sizeA = Inf;
B = fscanf(fileID,formatSpec,sizeA);
buf=ones(10,1,'double');
x=[B; buf];%Homogeneous form
x=reshape(x,10,3);
fclose(fileID);
A=zeros(20,12,'double');
j=1;
%Filling matrix A
for i=1:10
    A(j,[5:8])=-1*x(i,3)*X(i,[1:4]);
    A(j,[9:12])=x(i,2)*X(i,[1:4]);
    A(j+1,[1:4])=x(i,3)*X(i,[1:4]);
    A(j+1,[9:12])=-1*x(i,1)*X(i,[1:4]);
    j=j+2;
end
%Finding SVD to find P
[U,S,V]=svd(A,'econ');
P=-reshape(V([1:12],12),4,3);
P=P.';

x2=(P*X.').';%Projection of world points using matrix P
X2=zeros(10,1,'double');
X2=x2([1:10],3);
for i=1:3
    for j=1:10
        x2(j,i)=x2(j,i)/X2(j);
    end
end
x_actual_points=x
P_projection_matrix=P
x_points_projected_by_P=x2
[U,S,V]=svd(P);%To get nullspace vector of P
C_H=V([1:4],4);
C_homogeneous_form=C_H
for i=1:4;
    C_H(i)=C_H(i)/C_H(4);
end
C=C_H([1:3]);
C_null_space_of_P_inhomogeneous=C
[R Q]=rq(P(:,[1:3]));%RQ decomposition of matrix P for K*(R|-RC)
%We get RQ of -KR thats why only first three columns of P are taken
%R*Q=-KR
%-KRC is last column of P
%-inv(RQ)*P(last column)=C'
C_2=(-1*inv(R*Q)*P(:,4));%RQ decomposition of matrix 
C_from_RQ_decomposition_inhomogeneous=C_2



function [R Q]=rq(A)

[m n]=size(A);
if m>n
    error('RQ: Number of rows must be smaller than column');
end

[Q R]=qr(flipud(A).');
R=flipud(R.');
R(:,1:m)=R(:,m:-1:1);
Q=Q.';
Q(1:m,:)=Q(m:-1:1,:);

end
