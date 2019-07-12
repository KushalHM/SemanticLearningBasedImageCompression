# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:22:33 2019
"""

import cv2,os
from subprocess import call  

# Function to extract frames 
def FrameCapture(path,outFolde): 
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite(os.path.join(outFolde,"frame%d.jpg" % count), image)
        count += 1
  
def combineFrames(path,fileName):
    files = os.listdir(path)
    video = cv2.VideoWriter(fileName,cv2.VideoWriter_fourcc(*'XVID'),30,(640,360))
    for f in files:
        if (f == 'video'):
            continue
        video.write(cv2.imread(os.path.join(path,f)))
    cv2.destroyAllWindows()
    video.release()

# Driver Code 
if __name__ == '__main__': 
    # Calling the function 
    outFolde = "vidFrames"
    FrameCapture("vid.mp4",outFolde)
    print ("Frames extracted")
    combineFrames(outFolde, 'newVid.mp4')
    print ('Video combined without audio')
    i = 0;
    allFrames = os.listdir(outFolde)
    allFrames.sort()
    print ("Frame Count - ", len(allFrames))
    for file in allFrames:#[:25]:
        call("python generate_map.py "+os.path.join(outFolde,file), shell=True)
        call("python combine_images.py -image "+os.path.join(outFolde,file)+" -map "+os.path.join("output", "vidFrames",file) , shell=True)
        i +=1
        if (i % 10 == 0):
        	print ("Number of frames compressed ",i)

    print ("Video compressed")
    combineFrames(os.path.join("output",'video'), 'compressed.mp4')
    print ('Video compressed and combined')








    