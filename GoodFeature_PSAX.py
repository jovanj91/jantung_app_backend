import cv2
import os
import numpy as np

def high_boost_filter(image, lpf_image, kons):
    res = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lpf_rgb = lpf_image[i, j]
            src_rgb = image[i, j]

            for k in range(3):  # 3 channels (B, G, R)
                # val = kons * src_rgb[k] - lpf_rgb[k]
                val = kons * lpf_rgb[k]
                val = min(max(val, 0), 255)
                res[i, j, k] = val
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res

def morph(source):
    res = np.copy(source)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (3,3))
    res = cv2.morphologyEx(source, cv2.MORPH_OPEN, ellipse)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, ellipse)
    return res


def thresholding(source):
    res = np.copy(source)
    _, res = cv2.threshold(source, 10, 255, cv2.THRESH_BINARY)
    return res

def canny(source):
    res = np.copy(source)
    res = cv2.Canny(source, 0, 255, 3)
    return res

def region_filter(source):
    contours, hierarchy = cv2.findContours(source, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = np.zeros_like(source)
    for i in range(len(contours)):
        if len(contours[i]) > R:
            cv2.drawContours(res, contours, i, (255, 0, 0), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
           # cv2.drawContours(res, contours, i, (255, 0, 0), 1)
    return res

def coLinear(source):
    # Find contours in the input image
    contours, hierarchy = cv2.findContours(source, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = np.zeros_like(source)
    # data = [0] * len(contours)
    data = [0] * len(contours)

    idk = 0
    for i, contour in enumerate(contours):
        if len(contour) > R * 2:
            pt = contour[i][len(contour[i]) // 4]
            CCX[idk] = pt[0]
            CCY[idk] = pt[1]
            data[idk] = 0
        else:
            CCX[idk] = 0
            CCY[idk] = 0
            data[idk] = 1
        idk += 1
    # Intersection line evaluation
    for i in range(len(contours)):
        for j in range(len(contours)):
            if i == j: continue
            out = 0
            for k in range(len(contours[i]) // 2):
                pt1 = contours[i][k][0]
                pt2 = contours[i][k + 2][0]
                out = intersectionLine(X1, Y1, CCX[j], CCY[j], pt1[0], pt1[1], pt2[0], pt2[1])
                if out == 1:
                    if (abs(CCX[j] - pt1[0]) < 2) and (abs(CCY[j] - pt1[1]) < 2):
                        data[j] = 0
                    else:
                        data[j] = 1

    for i in range(len(contours)):
        if data[i] == 0:
            cv2.drawContours(res, contours, i, (255, 255, 255), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))

    return res
def intersectionLine(x1, y1, x2, y2, x3, y3, x4, y4):
    m1, c1 = straightLine(x1, y1, x2, y2)
    m2, c2 = straightLine(x3, y3, x4, y4)

    if m1 == m2:
        return 0

    xp = (c2 - c1) / (m1 - m2)
    yp = m1 * xp + c1

    if (x1 == x2) and ((xp - x3) * (xp - x4) < 0) and ((yp - y1) * (yp - y2) < 0):
        return 1
    if (x3 == x4) and ((xp - x1) * (xp - x2) < 0) and ((yp - y3) * (yp - y4) < 0):
        return 1

    if ((xp - x1) * (xp - x2) < 0) and ((xp - x3) * (xp - x4) < 0):
        return 1
    else:
        return 0

def straightLine(x1, y1, x2, y2):
    x = x1 - x2
    if x == 0:
        m = 1e6
    else:
        m = (y1 - y2) / x
    b = y1 - m * x1
    return m, b

# def slope(x1, y1, x2, y2):
#     tanx = (y2 - y1) / (x2 - x1)
#     s = math.atan(tanx)
#     s = (180 / math.pi) * s
#     return s

def GetGoodFeature(res):
    temp1, temp2, = 0, 0
    jumlah = 0
    banyak = 12

    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # minRect = []
    # for contour in contours:
    #     minRect.append(cv2.minAreaRect(contour))


    coordinate1 = []  # Create an empty list for storing coordinates

    for i in range(len(contours)):
        for j in range(len(contours[i]) // 2):
            jumlah += 1
            coordinate1.append(contours[i][j][0])

    temp1 = 0
    batasan = jumlah
    data1 = jumlah / (banyak + 1)

    coordinate2 = [None] * (banyak + 1)  # Initialize a list for coordinate2

    for i in np.arange(data1, batasan, data1):
        temp1 += 1
        temp2 = int(round(i))
        coordinate2[temp1] = coordinate1[temp2]

        if temp1 == banyak:
            break

    goodFeatures = []  # Create a list for storing good features
    for i in range(1, banyak + 1):
        goodFeatures.append(coordinate2[i])

    print(goodFeatures)
    frame_Draw = res
    for point in goodFeatures:
        cv2.circle(frame_Draw, (point[0], point[1]), 1, (255, 255, 255), 5, 8, 0)

    return frame_Draw


if __name__ == '__main__':
    R = 50
    X1, Y1 = 0, 0
    X2, Y2 = 0, 0
    CCX, CCY = [0] * 100, [0] * 100

    source = cv2.imread('frames/frame_0001.png')
    lpf = cv2.imread('medianfiltered/frame_0001.png')
    output_dir = 'hasil'
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, f'hasil.png')
    res = high_boost_filter(source, lpf, 1.5)
    cv2.imshow('hbf', res)
    res = morph(res)
    cv2.imshow('morph', res)
    res = thresholding(res)
    cv2.imshow('thres', res)
    res = canny(res)
    cv2.imshow('canny', res)
    res = region_filter(res)
    cv2.imshow('region', res)

    height, width = res.shape
    X1, Y1 = (width // 2), (height // 2)
    X2, Y2 = (X1 + 22), (Y1+ 23)

    res = coLinear(res)
    cv2.imshow('colinear', res)
    res = GetGoodFeature(res)
    cv2.imshow('hasil', res)
    cv2.waitKey(0)
    cv2.imwrite(output_image_path, res)