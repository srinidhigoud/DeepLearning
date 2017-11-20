


import cv2
import sys
import os.path
import numpy as np
import random

def is_invertible(a):
	return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
def compare(filename1, filename2):
    img1 = cv2.imread(filename1)          # queryImage
    img2 = cv2.imread(filename2)          # trainImage
    rows,cols,ch = img1.shape
    # pts1 = np.float32([[50,50],[200,50],[50,200]])
    # pts2 = np.float32([[10,100],[200,50],[100,250]])

    # M = cv2.getAffineTransform(pts1,pts2)
    # print(M)
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    img3 = cv2.drawKeypoints(img1,kp1,None,(255,255,0),4)
    img4 = cv2.drawKeypoints(img2,kp2,None,(255,255,0),4)
    cv2.imshow('with key points for image 1',img3)
    cv2.waitKey(0)
    cv2.destroyWindow('with key points for image 1')
    cv2.imshow('with key points for image 2',img4)
    cv2.waitKey(0)
    cv2.destroyWindow('with key points for image 2')
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    toogood=[]
    for m,n in matches:
    	if m.distance < 0.9*n.distance:
    		good.append([m])
    		toogood.append(m)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
    # print(kp1[1].pt) 
    # Show the image
    cv2.imshow('Matched Features', img3)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')
    src_pts = np.uint32([np.round((kp1[m.queryIdx]).pt) for m in toogood ]).reshape(-1,1,2)
    dst_pts = np.uint32([ np.round(kp2[m.trainIdx].pt) for m in toogood ]).reshape(-1,1,2)
    buf0,buf1,buf2=src_pts.shape
    buf1=buf0-1
    sum=0
    Y=[]
    Z=[]
    for l in range(0,100):
    	# 
	    (p1,p2,p3)=random.sample(range(0, buf0), 3)
	    A=np.array([src_pts[p1][0][0],src_pts[p1][0][1],0,0,1,0,0,0,src_pts[p1][0][0],src_pts[p1][0][1],0,1,src_pts[p2][0][0],src_pts[p2][0][1],0,0,1,0,0,0,src_pts[p2][0][0],src_pts[p2][0][1],0,1,src_pts[p3][0][0],src_pts[p3][0][1],0,0,1,0,0,0,src_pts[p3][0][0],src_pts[p3][0][1],0,1]).reshape(6,6)
	    B=np.array([dst_pts[p1][0][0],dst_pts[p1][0][1],dst_pts[p2][0][0],dst_pts[p2][0][1],dst_pts[p3][0][0],dst_pts[p3][0][1]]).reshape(6,1)
	    B=np.asarray(B)
	    A=np.asarray(A)
	    if(is_invertible(A)):
	    	p1=p1
	    else:
	    	continue

	    X=np.linalg.lstsq(A,B)[0]

	    if(l==0):
	    	Y.append(X)
	    M=np.array([X[0][0],X[1][0],X[2][0],X[3][0]]).reshape(2,2)
	    T=np.array([X[4][0],X[5][0]]).reshape(2,1)
	    z_local=[]
	    trpts=[]
	    sum_local=0
	    for i in range(0,buf0):
	    	 trpts.append(np.round(np.add(np.dot(M,(np.array(src_pts[i][:][0]))), T.T)))
	    	 if((np.linalg.norm(trpts[i][:][0]-dst_pts[i][:][0]))<=10):
	    	 	sum_local=sum_local+1
	    	 	z_local.append(i)
	    if(sum<sum_local):
	    	sum=sum_local
	    	Y[0]=X
	    	Z=np.array(z_local)
	    	
    A=[]
    B=[]
    alpha=0
    for it in Z:
    	alpha=alpha+1
    	A.append(np.array([src_pts[it][0][0],src_pts[it][0][1],0,0,1,0]))
    	A.append(np.array([0,0,src_pts[it][0][0],src_pts[it][0][1],0,1]))
    	B.append(np.array(dst_pts[it][0][0]))
    	B.append(np.array(dst_pts[it][0][1]))
    X1=np.linalg.lstsq(A,B)[0]
    avg=0
    
    
    H=np.array([X1[0],X1[1],X1[4],X1[2],X1[3],X1[5]]).reshape(2,3)
    print(H)
    dst = cv2.warpAffine(img1,H,(int(cols*1.5),rows))
    cv2.imshow('Input', img1)
    cv2.imshow('Input', img2)
    cv2.waitKey(0)
    cv2.imshow('Output', dst)
    cv2.waitKey(0)
    cv2.destroyWindow('Input')
    cv2.destroyWindow('Output')
    
if len(sys.argv) != 3:
    sys.stderr.write("usage: compare.py <queryImageFile> <sourceImageFile>\n")
    sys.exit(-1)
     
compare(sys.argv[1], sys.argv[2])