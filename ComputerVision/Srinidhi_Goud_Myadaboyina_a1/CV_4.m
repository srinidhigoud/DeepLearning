% clear all;
% close all;
% clc;
S=load('sfm_points.mat');
W=zeros(20,600);
t=zeros(10,2);
l=1;
X=zeros(2,600);
figure
%Just to see if the data will look like a cube finally
for k=1:10
    for i=1:600
        X([1:2],i)=S.image_points(:,i,k);
    end
    subplot(5,2,k)
    plot(X(1,:),X(2,:))
end
%Recentering the data
for i=1:10
    sum=zeros(2,1);
    for j=1:600
        y=S.image_points(:,j,i);
        sum=sum+y;
    end
    
    sum=sum/600;
    t(i,[1:2])=sum;%Matrix t
    for j=1:600
        W([l:l+1],j)=S.image_points(:,j,i)-sum;
    end
    l=l+2;
end
t
[U S V]=svd(W);
M_of_2x3_elements_for_10_images=U(:,[1:3])*S([1:3],[1:3])%Matrix M
V([1:10],[1:3])
rotate3d on
figure
plot3(V(:,1),V(:,2),V(:,3))