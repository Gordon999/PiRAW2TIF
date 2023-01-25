#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

# initial values
gamma_value = 2.2
contrast    = 1.5
saturation  = 1.6
brightness  = 1.0

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

# setup directories
Home_Files  = []
Home_Files.append(os.getlogin())

# find raw files
files = glob.glob("/home/" + Home_Files[0]+ "/Pictures/*.raw")
files.sort()
valid = 0

if len(files) > 0:
    for x in range(0,len(files)):
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
        # process if correct size
        if valid > 0:
            if image.size == 10171392:
                image = image.reshape(int(image.size/4128),4128)
                for j in range(4127,4099,-1):
                    image  = np.delete(image, j, 1)
            if image.size == 18580480:
                image = image.reshape(int(image.size/6112),6112)
                for j in range(6111,6083,-1):
                    image  = np.delete(image, j, 1)
            if image.size == 1586304:
                image = image.reshape(int(image.size/1632),1632)
                for j in range(1631,1619,-1):
                    image  = np.delete(image, j, 1)
            # extract data
            if valid == 1:
                A = image.reshape(int(image.size/5),5)
                A = np.split(A, [4,5],axis=1)
            else:
                A = image.reshape(int(image.size/3),3)
                A = np.split(A, [2,3],axis=1)
            F  = A[0].reshape(rows,cols)
            F = cv2.cvtColor(F, cv2.COLOR_BayerRG2BGR_EA)

            # adjust contrast
            lab= cv2.cvtColor(F, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8,8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl,a,b))
            F = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # adjust gamma
            F = gammaCorrection(F, gamma_value)

            # save output
            fname = files[x].split('.')
            cv2.imwrite(fname[0] + ".tif", F)
            
            # show result
            F = cv2.resize(F, dsize=(int(cols/4),int(rows/4)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('result',F)
    
        # wait for a key press
        cv2.waitKey()
    cv2.destroyAllWindows()

