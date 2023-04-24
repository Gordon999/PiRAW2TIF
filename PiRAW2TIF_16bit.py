#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

# v3.00

# setup directories
Home_Files  = []
Home_Files.append(os.getlogin())

# find raw files
files = glob.glob("/home/" + Home_Files[0]+ "/Pictures/*.raw")
files.sort()
valid = 0


if len(files) > 0:
    for x in range(5,len(files)):
        # Open raw file
        f = open(files[x],'rb')
        image = np.fromfile (f,dtype=np.uint8,count=-1)
        f.close()
        # check size
        if image.size == 1658880:  #Pi3 1536x864
            cols = 1536
            rows = 864
            valid = 1
        elif image.size == 14929920: #Pi3 4608x2592
            cols = 4608
            rows = 2592
            valid = 1
        elif image.size == 3732480: #Pi3 2304x1296
            cols = 2304
            rows = 1296
            valid = 1
        elif image.size == 384000:  #Pi2 640x480
            cols = 640
            rows = 480
            valid = 1
        elif image.size == 2562560: #Pi2 800x600
            cols = 1664
            rows = 1232
            valid = 1
        elif image.size == 10171392: #Pi2 3280x2464
            cols = 3280
            rows = 2464
            valid = 1
        elif image.size == 2592000:  #Pi2 1920x1080
            cols = 1920
            rows = 1080
            valid = 1
        elif image.size == 1586304:  #Pi1 1296x972
            cols = 1296
            rows = 972
            valid = 1
        elif image.size == 6345216:  #Pi1 2592x1944
            cols = 2600
            rows = 1944
            valid = 1
        elif image.size == 4669440:  #PiHQ 2028x1520
            cols = 2048
            rows = 1520
            valid = 2
        elif image.size == 3317760:  #PiHQ 2028x1080
            cols = 2048
            rows = 1080
            valid = 2
        elif image.size == 18580480:  #PiHQ 4056x3040
            cols = 4056
            rows = 3040
            valid = 2
        else:
            valid = 0
            print("Failed to find suitable file ",files[x])
   
        # process if a valid size
        if valid > 0:
            # trim off 
            if image.size == 10171392:
                image = image.reshape(int(image.size/4128),4128)
                for j in range(4127,4099,-1):
                    image  = np.delete(image, j, 1)
            elif image.size == 6345216:
                image = image.reshape(int(image.size/3264),3264)
                for j in range(3263,3249,-1):
                    image  = np.delete(image, j, 1)
            elif image.size == 18580480:
                image = image.reshape(int(image.size/6112),6112)
                for j in range(6111,6083,-1):
                    image  = np.delete(image, j, 1)
            elif image.size == 1586304:
                image = image.reshape(int(image.size/1632),1632)
                for j in range(1631,1619,-1):
                    image  = np.delete(image, j, 1)
            # extract data
            if valid == 1:
                A = image.reshape(int(image.size/5),5)
                B  = np.split(A, [4,5], axis=1)
            else:
                A = image.reshape(int(image.size/3),3)
                B  = np.split(A, [2,3], axis=1)

            B[0] = B[0] * 256
            C  = B[0].reshape(int(rows/2),int(cols*2))
            D  = np.split(C, 2, axis=1)
            H  = D[0].reshape(int(D[0].size/2),2)
            I  = np.split(H, 2, axis=1)
            if valid == 1:
                I[0] = I[0] + (np.unpackbits(B[1], axis=1)[:,0:1]*128) + (np.unpackbits(B[1], axis=1)[:,1:2]*64)
                I[1] = I[1] + (np.unpackbits(B[1], axis=1)[:,2:3]*128) + (np.unpackbits(B[1], axis=1)[:,3:4]*64)
            if valid == 2:
                E  = B[1].reshape(int(B[1].size/2),2)
                F  = np.split(E, 2, axis=1)
                I[0] = I[0] + (np.unpackbits(F[0], axis=1)[:,0:1]*128) + (np.unpackbits(F[0], axis=1)[:,1:2]*64) + (np.unpackbits(F[0], axis=1)[:,2:3]*32) + (np.unpackbits(F[0], axis=1)[:,3:4]*16)
                I[1] = I[1] + (np.unpackbits(F[1], axis=1)[:,4:5]*128) + (np.unpackbits(F[1], axis=1)[:,5:6]*64) + (np.unpackbits(F[1], axis=1)[:,6:7]*32) + (np.unpackbits(F[1], axis=1)[:,7:8]*16)
            b  = I[0].reshape(int(rows/2),int(cols/2))
            g0 = I[1].reshape(int(rows/2),int(cols/2))
            L  = D[1].reshape(int(D[0].size/2),2)
            M  = np.split(L, 2, axis=1)
            if valid == 1:
                M[0] = M[0] + (np.unpackbits(B[1], axis=1)[:,4:5]*128) + (np.unpackbits(B[1], axis=1)[:,5:6]*64)
                M[1] = M[1] + (np.unpackbits(B[1], axis=1)[:,6:7]*128) + (np.unpackbits(B[1], axis=1)[:,7:8]*64)
            if valid == 2:
                E  = B[1].reshape(int(B[1].size/2),2)
                F  = np.split(E, 2, axis=1)
                M[0] = M[0] + (np.unpackbits(F[0], axis=1)[:,0:1]*128) + (np.unpackbits(F[0], axis=1)[:,1:2]*64) + (np.unpackbits(F[0], axis=1)[:,2:3]*32) + (np.unpackbits(F[0], axis=1)[:,3:4]*16)
                M[1] = M[1] + (np.unpackbits(F[1], axis=1)[:,4:5]*128) + (np.unpackbits(F[1], axis=1)[:,5:6]*64) + (np.unpackbits(F[1], axis=1)[:,6:7]*32) + (np.unpackbits(F[1], axis=1)[:,7:8]*16)
            g1 = M[0].reshape(int(rows/2),int(cols/2))
            r  = M[1].reshape(int(rows/2),int(cols/2))

            # some basic colour correction
            Red   = r * 1
            Blue  = b * 1
            Green = ((g0/2) + (g1/2)) * 0.7
            Green = Green.astype(np.uint16)

            # combine B,G,R
            BGR=np.dstack((Blue,Green,Red)).astype(np.uint16)
            res = cv2.resize(BGR, dsize=(cols,rows), interpolation=cv2.INTER_CUBIC)
            res = res.astype(np.uint16)
                 
            # save output
            fname = files[x].split('.')
            cv2.imwrite(fname[0] + ".tif", res)

            # show corrected result
            result = cv2.resize(res, dsize=(int(cols/4),int(rows/4)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('Output',result)
            
    
    
        # wait for a key press
        cv2.waitKey()
    cv2.destroyAllWindows()
