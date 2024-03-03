import cv2
import os
import numpy as np
import math

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
    _, res = cv2.threshold(source, 25, 255, cv2.THRESH_BINARY)
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
    coordinate1 = [[(0, 0) for _ in range(10)] for _ in range(500)]
    coordinate2 = [[(0, 0) for _ in range(10)] for _ in range(500)]
    temp1, temp2, temp3 = 0, 0, 0
    rect_points = np.zeros((4, 2), dtype=np.float32)
    color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
    garis = np.zeros(res.shape, dtype=res.dtype)
    hasil = np.zeros(res.shape, dtype=res.dtype)

    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    minRect = [cv2.minAreaRect(contour) for contour in contours]

    for i in range(len(contours)):
        cv2.drawContours(garis, contours, i, color)
        rect_points = cv2.boxPoints(minRect[i]).astype(int)

    print('rect : ' + str(rect_points))
    cv2.polylines(garis, [rect_points], True, color, 2)
    cv2.imshow('garis', garis)
    # kondisi1 = rect_points[3][0] - rect_points[0][0]
    # kondisi2 = rect_points[0][1] - rect_points[3][1]

    kondisi1 = rect_points[2][0] - rect_points[1][0]
    kondisi2 = rect_points[1][1] - rect_points[2][1]

    print("kondisi 1 :" + str(kondisi1))
    print("kondisi 2 :" + str(kondisi2))

    if kondisi1 < kondisi2:
        valnorm = math.sqrt(pow(rect_points[1][0] - rect_points[2][0], 2) + pow(rect_points[1][1] - rect_points[2][1], 2))

        print('kanan')

        # garis kanan
        garis = np.zeros(res.shape, dtype=res.dtype)
        cv2.line(garis, (rect_points[2]), (rect_points[3]), color)

        for y in range(garis.shape[0]):
            for x in range(garis.shape[1]):
                if garis[y, x] > 0:
                    temp1 += 1
                    coordinate1[temp1][0] = (x, y)

        batasan = temp1
        data = float(temp1) / (jumlah + 1)
        temp1 = 0
        for i in np.arange(data / 2, batasan, data):
            temp1 += 1
            temp2 = int(round(i))
            coordinate2[temp1][0] = coordinate1[temp2][0]
            if temp1 == jumlah:
                break

        # Garis kiri
        garis = np.zeros(res.shape, dtype=res.dtype)
        temp1 = 0
        temp2 = 0
        cv2.line(garis, (rect_points[0]), (rect_points[1]), color)
        for y in range(garis.shape[0]):
            for x in range(garis.shape[1]):
                if garis[y, x] > 0:
                    temp1 += 1
                    coordinate1[temp1][0] = (x, y)

        batasan = temp1
        data = float(temp1) / (jumlah + 1)
        temp1 = 0
        for i in np.arange(data / 2, batasan, data):
            temp1 += 1
            temp2 = int(round(i))
            coordinate2[temp1][0] = coordinate1[temp2][0]
            if temp1 == jumlah:
                break

        garis = np.zeros(res.shape, dtype=res.dtype)
        temp1 = 0
        temp2 = jumlah

        for i in range(jumlah):
            cv2.line(garis, int(coordinate2[i][0]), int(coordinate2[i][1]), 255, 1, 1, 0)
            hasil = cv2.bitwise_and(garis, res)
            temp3 = 0
            for x in range(hasil.shape[1] - 1, 0, -1):
                for y in range(hasil.shape[0] - 1, 0, -1):
                    if hasil[y, x] > 0:
                        temp1 += 1
                        temp3 += 1
                        coordinate2[temp1][2] = (x, y)
                        break
                if temp3 > 0:
                    break

            for x in range(hasil.shape[1]):
                for y in range(hasil.shape[0]):
                    if hasil[y, x] > 0:
                        temp2 += 1
                        coordinate2[temp2][2] = (x, y)
                        hasil = np.zeros(res.shape, dtype=res.dtype)
                        garis = np.zeros(res.shape, dtype=res.dtype)
        return coordinate2

    else:
        valnorm = np.sqrt((rect_points[2][0] - rect_points[3][0]) ** 2 + (rect_points[2][1] - rect_points[3][1]) ** 2)
        print('kiri')

        #garis kanan
        garis = np.zeros(res.shape, dtype=res.dtype)
        cv2.line(garis, (rect_points[0]), (rect_points[3]), (255, 255, 255))
        cv2.imshow('intersect', garis)
        for x in range(garis.shape[1]):
            for y in range(garis.shape[0]):
                if garis[y, x] > 0:
                    temp1 += 1
                    coordinate1[temp1][0] = (x, y)

        batasan = temp1
        data = float(temp1) / (jumlah + 1)
        temp1 = 0

        for i in range(int(data / 2), int(batasan) + 1, int(data)):
            temp1 += 1
            temp2 = int(round(i))
            coordinate2[temp1][0] = coordinate1[temp2][0]
            if temp1 == jumlah:
                break

        #garis kiri
        temp1 = 0
        garis = np.zeros(res.shape, dtype=res.dtype)
        cv2.line(garis, (rect_points[1]), (rect_points[2]), (255, 255, 255))
        cv2.imshow('sisi kanan', garis)
        for x in range(garis.shape[1]):
            for y in range(garis.shape[0]):
                if garis[y, x] > 0:
                    temp1 += 1
                    coordinate1[temp1][1] = (x, y)

        batasan = temp1
        data = float(temp1) / (jumlah + 1)
        temp1 = 0
        temp2 = 0

        for i in range(int(data / 2), int(batasan) + 1, int(data)):
            temp1 += 1
            temp2 = int(round(i))
            coordinate2[temp1][1] = coordinate1[temp2][1]
            if temp1 == jumlah:
                break

        temp1 = 0
        temp2 = jumlah
        garis = np.zeros(res.shape, dtype=res.dtype)

        for i in range(1, jumlah + 1):
            cv2.line(garis, coordinate2[i][0], coordinate2[i][1], 255, 1, 1, 0)
            hasil = cv2.bitwise_and(garis, res)
            temp3 = 0

            for x in range(hasil.shape[1] - 1, 0, -1):
                for y in range(hasil.shape[0] - 1, 0, -1):
                    if hasil[y, x] > 0:
                        temp1 += 1
                        temp3 += 1
                        coordinate2[temp1][2] = (x, y)
                        break
                if temp3 > 0:
                    break

            for x in range(hasil.shape[1]):
                for y in range(hasil.shape[0]):
                    if hasil[y, x] > 0:
                        temp2 += 1
                        coordinate2[temp2][2] = (x, y)
                        hasil = np.zeros(res.shape, dtype=res.dtype)
                        garis = np.zeros(res.shape, dtype=res.dtype)
                        break


        return coordinate2

def find_angle(x1, y1, x2, y2):
    angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

    if -90 <= angle < 0:
        angle = abs(angle) + 90
    elif 0 <= angle < 90:
        angle = angle - 180 + 90
        angle = abs(angle) + 360
    elif 90 <= angle <= 180:
        angle = -(angle - 180) + 90
        angle += 180
    else:
        angle = abs(angle) + 90

    return angle

# def optical_flow_calc():


if __name__ == '__main__':
    R = 10 #radius contour
    X1, Y1 = 0, 0 #titik tengah gambar
    X2, Y2 = 0, 0
    CCX, CCY = [0] * 100, [0] * 100

    jumlah = 12
    goodFeatures = [[] for _ in range(10)]
    GFcoordinate = [[(0, 0) for _ in range(10)] for _ in range(500)]

    tresh_diff = 20.0
    length = []*4
    term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03)
    win_size = (50, 50)

    source = cv2.imread('1.frames/frame_0000.png')
    lpf = cv2.imread('2.medianfiltered/median.png')
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

    GFcoordinate = GetGoodFeature(res)
    for i in range(1, jumlah * 2 + 1):
        x = GFcoordinate[i][2][0]
        y = GFcoordinate[i][2][1]
        goodFeatures[0].append((x, y))

    print(goodFeatures)

    #Visualisasi Good Feature
    for i in range(jumlah*2):
        x, y = goodFeatures[0][i]
        print("i" + str(x,y))
        cv2.circle(res, (int(x), int(y)), 1, (255, 255, 255), 2, 8, 0)

    cv2.imshow('GoodFeature', res)
    cv2.waitKey(0)
    cv2.imwrite(output_image_path, res)