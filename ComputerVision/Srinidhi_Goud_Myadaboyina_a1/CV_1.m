close all;
clear all;
clc;

%1)2 part c

filtk=[1/4 2/4 1/4];
filt2k=filtk;
prompt = 'What is the width of the kernel? ';
x=input(prompt);
prompt='Now select the image, after navigating to its folder';

while(mod(x,2)==0)%Enter only odd kernel width
    prompt = 'Enter odd value? ';
    x=input(prompt);
end
prompt='Now select the image, after navigating to its folder. Type any integer and press return/enter to continue';
y=input(prompt);
%You need to enter any integer and press enter and then select any image, that will be feeded into the system
sz=fix(x/2)-1;
for i=1:sz
    filt2k=conv2(filtk,filt2k);%Convolve for the given width
end
image=uigetfile(['*.*']);
RGB = (imread(image));
RGB2=imfilter(RGB,filt2k);%Convolving in x-axis
blurred=imfilter(RGB2.',filt2k);%Convolving in y-axis
blurred=blurred.';
figure
subplot(2,1,1)
imshow(RGB);
title('Original Image');

subplot(2,1,2);
imshow(blurred);
title(['Blurred Image with width=',num2str(x)]);
