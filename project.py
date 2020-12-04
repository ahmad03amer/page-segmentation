import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# Read The input image
path = "venv/Resources/00000126.tif"
img = cv2.imread(path)
imgDraw = cv2.imread(path)

# Convert the image to gray scale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blureing the image using kernel
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

# Apply OTSU threshold
ret, thresh1 = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

# declare a kernel
kernel = np.ones((3, 4), np.int8)

# Apply erosion
img_erosion = cv2.erode(thresh1, kernel, iterations=5)

# Apply dilation
img_dilation = cv2.dilate(thresh1, kernel, iterations=9)

blur2 = cv2.GaussianBlur(img_erosion, (1, 1), 0)

# draw a rectangle around the paragraph
cnts = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

# select the paragraphs in hand line
cntsDraw = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cntsDraw = cntsDraw[0] if len(cntsDraw) == 2 else cntsDraw[1]

for c in cntsDraw:
    cv2.drawContours(imgDraw, c, -1, (255, 0, 0), 4)


# Resize the image
imgResult = cv2.resize(img, (int(img.shape[1]*0.3), int(img.shape[0]*0.3)))
imgDraw = cv2.resize(imgDraw, (int(imgDraw.shape[1]*0.3), int(imgDraw.shape[0]*0.3)))

# join more than one figure with many channel
imgStack = stackImages(0.4, ([img, imgDraw, imgBlur],[img_erosion,img_dilation,imgGray]))

#cv2.imshow("Stack Output", imgStack)
# save copy of the result

cv2.imwrite("00000126.tif",imgDraw)

#cv2.imshow('Output Rectangle', imgResult)

cv2.imshow('Output draw', imgDraw)

cv2.waitKey(0)