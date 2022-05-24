import os
import io
from unittest import result
import cv2
import requests
from enum import Enum
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

from PIL import Image,ImageFilter
from imutils.object_detection import non_max_suppression

import numpy as np
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="sqy-watermark-engine",
    description="Use this API to paste Square Yards logo as a watermark at the center of input images",
    version="1.0",
)

class URL2(BaseModel):
    url_: str
    width_percentage: Optional[float] = 0.3
    
@app.get("/")
async def root():
    return "Hello World!!!"

@app.get("/all_watermark_removal")
async def removal(watermark_image: str):
    
    parsed = urlparse(watermark_image)
    # print(os.path.basename(parsed.path))
    
    response = requests.get(watermark_image)
    image_bytes = io.BytesIO(response.content)
    
    original_image = Image.open(image_bytes)

    
    format_1 = original_image.format.lower()
    filename = watermark_image 


    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")):

        original_image = Image.open(image_bytes)
        #this function get the format type of input image
        def get_format(filename):
            format_ = filename.split(".")[-1]
            if format_.lower() == "jpg":
                format_ = "jpeg"
            elif format_.lower == "webp":
                format_ = "WebP"
        
            return format_
    
    
        #this function for gave the same type of format to output
        def get_content_type(format_):
            type_ = "image/jpeg"
            if format_ == "gif":
                type_ = "image/gif"
            elif format_ == "webp":
                type_ = "image/webp"
            elif format_ == "png":
                type_ = "image/png"
            #print(type_)
            return type_

        format_ = get_format(filename)#format_ store the type of image by filename

        img_save = original_image.save("detect_img.jpeg")
        image1 = cv2.imread("detect_img.jpeg",cv2.IMREAD_COLOR) 
        # im_path = "api/results/result13.jpg"
        image1 = cv2.resize(image1, (image1.shape[1],image1.shape[0]))
        ima_org = image1.copy()

        (height1, width1) = image1.shape[:2]

        size = 640 #size must be multiple of 32. Haven't tested with smaller size which can increase speed but might decrease accuracy.
        (height2, width2) = (size, size)  
        image2 = cv2.resize(image1, (width2, height2)) 
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        blob = cv2.dnn.blobFromImage(image2, 1.0, (width2, height2), (103.68, 100.78, 50.94), swapRB=True, crop=False)
        net.setInput(blob)

        (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        (rows, cols) = scores.shape[2:4]  # grab the rows and columns from score volume
        rects = []  # stores the bounding box coordiantes for text regions
        confidences = []  # stores the probability associated with each bounding box region in rects

        for y in range(rows):
            scoresdata = scores[0, 0, y]
            xdata0 = geometry[0, 0, y]
            xdata1 = geometry[0, 1, y]
            xdata2 = geometry[0, 2, y]
            xdata3 = geometry[0, 3, y]
            angles = geometry[0, 4, y]

            for x in range(cols):

                if scoresdata[x] < 0.5:  # if score is less than min_confidence, ignore
                    continue
                # print(scoresdata[x])
                offsetx = x * 4.0
                offsety = y * 4.0
                # EAST detector automatically reduces volume size as it passes through the network
                # extracting the rotation angle for the prediction and comput ing their sine and cos

                angle = angles[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xdata0[x] + xdata2[x]
                w = xdata1[x] + xdata3[x]
                # print(offsetx,offsety,xdata1[x],xdata2[x],cos)
                endx = int(offsetx + (cos * xdata1[x]) + (sin * xdata2[x]))
                endy = int(offsety + (sin * xdata1[x]) + (cos * xdata2[x]))
                startx = int(endx - w)
                starty = int(endy - h)

                # appending the confidence score and probabilities to list
                rects.append((startx, starty, endx, endy))
                confidences.append(scoresdata[x])

        # applying non-maxima suppression to supppress weak and overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        iti=[]
        rW = width1 / float(width2)
        rH = height1 / float(height2)


        bb = []

        for (startx, starty, endx, endy) in boxes:
            startx = int(startx * rW)
            starty = int(starty * rH)
            endx = int(endx * rW)
            endy = int(endy * rH)
            cv2.rectangle(image1, (startx, starty), (endx, endy), (255, 0,0), 2)
            
            bb.append([startx, starty, endx, endy])

        original_image.save("imgs.jpeg")
        path_im = "imgs.jpeg"
        im = cv2.imread("imgs.jpeg")

        img = np.zeros(im.shape,dtype=np.uint8)
        img.fill(0) # or img[:] = 255
        im = Image.fromarray(img) #convert numpy array to image
        im.save('white.jpg')

        image = Image.open(path_im)
        im = Image.open("white.jpg")

        for i in range (len(bb)):

            im_c = image.crop(bb[i])
            im.paste(im_c,bb[i])   
            i+=1

        im.save("mask.png")
        original_image = Image.open(image_bytes)
        original_image.save("image."+format_1)

        img = cv2.imread("image."+format_1)
        mask = cv2.imread('mask.png',0)
        dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

        cv2.imwrite(os.path.basename(parsed.path)+"."+format_1.lower(),dst)
        image1 = Image.open(os.path.basename(parsed.path)+"."+format_1.lower())
        filename1 = (os.path.basename(parsed.path))
        buffer = BytesIO()
        image1.save(buffer, format=format_, quality=100)
        buffer.seek(0)

        return StreamingResponse(buffer, media_type=get_content_type(format_),headers={'Content-Disposition': 'inline; filename="%s"' %(filename1,)})

    