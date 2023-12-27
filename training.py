import os, datetime, werkzeug, copy
import numpy as np
import cv2
import math

R = 30 #radius
X1, Y1 = 0, 0 #centerpoint
X2, Y2 = 0, 0
CCX, CCY = [0] * 100, [0] * 100

jumlah = 6
goodFeatures = [np.array([[]]) for _ in range(10)]
GFcoordinates = {}
valnorm = 0

lengthDif = [[] for _ in range (9)]

direction = np.zeros((12, 9), dtype=float)
directionI = np.zeros((12, 9), dtype=float)

jumlahFrame = 10
frames = {}
res ={}

def video2frames(video):
    rawImages = {}
    cap = cv2.VideoCapture(video)
    target_frames = jumlahFrame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(total_frames // target_frames, 1)
    if total_frames <= target_frames:
        frame_skip = 1
    else:
        frame_skip = total_frames // target_frames
    frame_count = 0
    frame_index = 0
    while frame_index < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            rawImages[frame_count] = frame
            frame_index += 1
        frame_count += 1
    cap.release()
    return rawImages

def median_filter(image):
    res = np.copy(image)
    kernelsize = 27
    res = cv2.medianBlur(image, kernelsize)
    return res

def high_boost_filter( image, lpf, kons):
    res = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lpf_rgb = lpf[i, j]
            src_rgb = image[i, j]
            for k in range(3):  # 3 channels (B, G, R)
                # val = kons * src_rgb[k] - lpf_rgb[k]
                val = kons * lpf_rgb[k]
                val = min(max(val, 0), 255)
                res[i, j, k] = val
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return res


def morph(image):
    res = np.copy(image)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (3,3))
    res = cv2.morphologyEx(image, cv2.MORPH_OPEN, ellipse)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, ellipse)
    return res

def thresholding( image):
    res = np.copy(image)
    _, res = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY) #original at 90
    return res

def canny( image):
    res = image.copy()
    res = cv2.Canny(image, 0, 255, 3)
    return res

def region_filter(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = np.zeros_like(image)
    for i in range(len(contours)):
        if len(contours[i]) > R:
            # cv2.drawContours(res, contours, i, (255, 0, 0), 1)
            cv2.drawContours(res, contours, i, (255, 0, 0), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
    return res


def coLinear( image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = np.zeros_like(image)
    data = [0] * 100

    idk = 0
    for i in range(len(contours)):
        if len(contours[i]) > R * 2:
            pt = contours[i][len(contours[i]) // 4][0]

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

def intersectionLine( x1, y1, x2, y2, x3, y3, x4, y4):
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

def straightLine( x1, y1, x2, y2):
    x = x1 - x2
    if x == 0:
        m = 1e6
    else:
        m = (y1 - y2) / x
    b = y1 - m * x1
    return m, b

def GetGoodFeaturesPSAX( res):
    temp1, temp2, = 0, 0
    count = 0
    banyak = jumlah * 2

    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # minRect = []
    # for contour in contours:
    #     minRect.append(cv2.minAreaRect(contour))


    coordinate1 = []  # Create an empty list for storing coordinates

    for i in range(len(contours)):
        for j in range(len(contours[i]) // 2):
            count += 1
            coordinate1.append(contours[i][j][0])

    temp1 = 0
    batasan = count
    data1 = count / (banyak + 1)

    coordinate2 = [None] * (banyak + 1)

    for i in np.arange(data1, batasan, data1):
        temp1 += 1
        temp2 = int(round(i))
        coordinate2[temp1] = coordinate1[temp2]

        if temp1 == banyak:
            break

    goodFeatures = []  # Create a list for storing good features
    for i in range(1, banyak + 1):
        goodFeatures.append(coordinate2[i])

    return goodFeatures

def GetGoodFeaturesIntersection( res):
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
        rect_points = cv2.boxPoints(minRect[i])

    kondisi1 = rect_points[3][0] - rect_points[0][0]
    kondisi2 = rect_points[0][1] - rect_points[3][1]

    if kondisi1 < kondisi2:
        print('kanan')
        valnorm = np.sqrt((rect_points[2][0] - rect_points[3][0]) ** 2 + (rect_points[2][1] - rect_points[3][1]) ** 2)

        # garis kanan
        garis = np.zeros(res.shape, dtype=res.dtype)
        cv2.line(garis, (int(rect_points[3][0]), int(rect_points[3][1])), (int(rect_points[0][0]), int(rect_points[0][1])), color)

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
        cv2.line(garis, (int(rect_points[1][0]), int(rect_points[1][1])), (int(rect_points[2][0]), int(rect_points[2][1])), color)
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
        valnorm = np.sqrt((rect_points[1][0] - rect_points[2][0]) ** 2 + (rect_points[1][1] - rect_points[2][1]) ** 2)
        print('kiri')

        garis = np.zeros(res.shape, dtype=res.dtype)
        cv2.line(garis, (int(rect_points[2][0]), int(rect_points[2][1])), (int(rect_points[3][0]), int(rect_points[3][1])), color)

        for x in range(garis.shape[1]):
            for y in range(garis.shape[0]):
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

        temp1 = 0
        garis = np.zeros(res.shape, dtype=res.dtype)
        cv2.line(garis, (int(rect_points[0][0]), int(rect_points[0][1])), (int(rect_points[1][0]), int(rect_points[1][1])), color)

        for x in range(garis.shape[1]):
            for y in range(garis.shape[0]):
                if garis[y, x] > 0:
                    temp1 += 1
                    coordinate1[temp1][1] = (x, y)

        batasan = temp1
        data = float(temp1) / (jumlah + 1)
        temp1 = 0
        temp2 = 0

        for i in np.arange(data / 2, batasan, data):
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

def findAngle(x1, y1, x2, y2):
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


def opticalFlowCalcwithNormalization(sources, goodFeatures):
    thresh_diff = 20.0
    termCrit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03)
    winSize = (50, 50)
    length = [[] for _ in range(4)]
    for i in range(9):
        maxLevel = 3
        sources[i] = cv2.medianBlur(sources[i], 9)
        #cv2.calcOpticalFlowPyrLK(sources[i], sources[i + 1], goodFeatures[i], goodFeatures[i + 1], status, errs[i], winSize, maxLevel, termCrit)
        goodFeatures[i + 1], status, errs = cv2.calcOpticalFlowPyrLK(sources[i], sources[i + 1], goodFeatures[i], winSize, maxLevel, termCrit)
        print(status[i])
        print(errs[i])

        for k in range(4):
            for j in range(len(goodFeatures[i])):
                length[0] = np.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j][0][0]) ** 2 + (goodFeatures[i][j][0][1] - goodFeatures[i + 1][j][0][1]) ** 2) / valnorm * 100
                if length[0] > thresh_diff:
                    if (j > 0 and j < 5) or (j > 6 and j < 11):
                        length[1] = np.sqrt((goodFeatures[i][j - 1][0][0] - goodFeatures[i + 1][j - 1][0][0]) ** 2 + (goodFeatures[i][j - 1][0][1] - goodFeatures[i + 1][j - 1][0][1]) ** 2) / valnorm * 100
                        length[2] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2) / valnorm * 100

                        if length[1] < thresh_diff:
                            length[3] = np.sqrt((goodFeatures[i][j - 1][0][0] - goodFeatures[i + 1][j - 1][0][0]) ** 2 + (goodFeatures[i][j - 1][0][1] - goodFeatures[i + 1][j - 1][0][1]) ** 2)
                            angleNorm = findAngle(goodFeatures[i][j - 1][0][0], goodFeatures[i][j - 1][0][1], goodFeatures[i + 1][j - 1][0][0], goodFeatures[i + 1][j - 1][0][1])
                            s = np.sin(angleNorm * np.pi / 180)
                            c = np.cos(angleNorm * np.pi / 180)
                            P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                            goodFeatures[i + 1][j][0] = P

                        elif length[2] < thresh_diff:
                            length[3] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2)
                            angleNorm = findAngle(goodFeatures[i][j + 1][0][0], goodFeatures[i][j + 1][0][1], goodFeatures[i + 1][j + 1][0][0], goodFeatures[i + 1][j + 1][0][1])
                            s = np.sin(angleNorm * np.pi / 180)
                            c = np.cos(angleNorm * np.pi / 180)
                            P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                            goodFeatures[i + 1][j][0] = P

            for j in range(len(goodFeatures[i])):
                length[0] = np.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j][0][0]) ** 2 + (goodFeatures[i][j][0][1] - goodFeatures[i + 1][j][0][1]) ** 2) / valnorm * 100
                if length[0] > thresh_diff:
                    if j == 0 or j == 6:
                        length[3] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2)
                        angleNorm = findAngle(goodFeatures[i][j + 1][0][0], goodFeatures[i][j + 1][0][1], goodFeatures[i + 1][j + 1][0][0], goodFeatures[i + 1][j + 1][0][1])
                        s = np.sin(angleNorm * np.pi / 180)
                        c = np.cos(angleNorm * np.pi / 180)
                        P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                        goodFeatures[i][j][0] = P

                    elif j == 5 or j == 11:
                        length[3] = np.sqrt((goodFeatures[i][j - 1][0][0] - goodFeatures[i + 1][j - 1][0][0]) ** 2 + (goodFeatures[i][j - 1][0][1] - goodFeatures[i + 1][j - 1][0][1]) ** 2)
                        angleNorm = findAngle(goodFeatures[i][j - 1][0][0], goodFeatures[i][j - 1][0][1], goodFeatures[i][j - 1][0][0], goodFeatures[i][j - 1][0][1])
                        s = np.sin(angleNorm * np.pi / 180)
                        c = np.cos(angleNorm * np.pi / 180)
                        P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                        goodFeatures[i][j][0] = P

    for i in range(9):
        for j in range(len(goodFeatures[i])):
            length = (math.sqrt(((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j][0][0]) ** 2) + ((goodFeatures[i][j][0][1] - goodFeatures[i + 1][j][0][1]) ** 2)) / valnorm) * 100
            lengthDif[i].append(length)

def opticalFlowCalc( sources, goodFeatures):
    termCrit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03)
    winSize = (50, 50)
    for i in range(len(sources)):
        sources[i] = cv2.cvtColor(sources[i], cv2.COLOR_BGR2GRAY)
    for i in range(len(sources) - 1):
        maxLevel = 3
        sources[i] = cv2.medianBlur(sources[i], 9)
        #cv2.calcOpticalFlowPyrLK(sources[i], sources[i + 1], goodFeatures[i], goodFeatures[i + 1], status, errs[i], winSize, maxLevel, termCrit)
        goodFeatures[i + 1], status, errs = cv2.calcOpticalFlowPyrLK(sources[i], sources[i + 1], goodFeatures[i], winSize, maxLevel, termCrit)
        print(status[i])
        print(errs[i])


    for i in range(len(sources)-1):
        for j in range(jumlah * 2):
            output_path = os.path.join('Tracking', f'TrackingLK.png')
            gfx_awal = int(goodFeatures[0][j][0][0])
            gfy_awal = int(goodFeatures[0][j][0][1])
            gfx_akhir = int(goodFeatures[len(sources) - 1][j][0][0])
            gfy_akhir = int(goodFeatures[len(sources) - 1][j][0][1])
            cv2.line(sources[0], (gfx_awal, gfy_awal), (gfx_akhir, gfy_akhir), (255, 255, 255), 1)
            cv2.imwrite(output_path, sources[0])

            length = math.sqrt((gfx_awal - gfx_akhir)**2 + (gfy_awal - gfy_akhir)**2)
            lengthDif[i].append(length)

def featureExtraction(goodFeatures):
    for j in range(jumlah):
        for i in range(9):
            # PENCARIAN SISI KIRI(GOODFEATURE) DERAJAT KEMIRINGAN
            a1 = math.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j + jumlah][0][0]) ** 2 + (goodFeatures[i][j][0][1] - goodFeatures[i + 1][j + jumlah][0][1]) ** 2)
            b1 = math.sqrt((goodFeatures[i + 1][j + jumlah][0][0] - goodFeatures[i][j + jumlah][0][0]) ** 2 +
                            (goodFeatures[i + 1][j + jumlah][0][1] - goodFeatures[i][j + jumlah][0][1]) ** 2)
            c1 = math.sqrt((goodFeatures[i][j + jumlah][0][0] - goodFeatures[i][j][0][0]) ** 2 +
                            (goodFeatures[i][j + jumlah][0][1] - goodFeatures[i][j][0][1]) ** 2)
            angle1 = math.acos((b1 ** 2 + c1 ** 2 - a1 ** 2) / (2 * b1 * c1)) * 180 / math.pi
            quadrant1 = (b1 ** 2 + c1 ** 2 - a1 ** 2) / (2 * b1 * c1) * 180 / math.pi
            if quadrant1 >= -1.27222e-14:
                # MASUK
                direction[j + jumlah][i] = int(1)
                slope1 = (goodFeatures[i + 1][j + jumlah][0][1] - goodFeatures[i][j + jumlah][0][1]) / (
                    goodFeatures[i + 1][j + jumlah][0][0] - goodFeatures[i][j + jumlah][0][0])
                slope2 = (goodFeatures[i][j][0][1] - goodFeatures[i][j + jumlah][0][1]) / (
                    goodFeatures[i][j][0][0] - goodFeatures[i][j + jumlah][0][0])
                if slope1 > slope2:
                    print("MASUK ++", angle1)
                    directionI[j + jumlah][i] = int(1)
                else:
                    print("MASUK --", angle1)
                    directionI[j + jumlah][i] = int(2)
                # PEMBAGIAN EKSTRAKSI DILAKUKAN DISINI YA UNTUK BAGIAN YANG MASUK
            else:
                # KELUAR
                direction[j + jumlah][i] = int(0)
                slope1 = (goodFeatures[i + 1][j + jumlah][0][1] - goodFeatures[i][j + jumlah][0][1]) / (
                        goodFeatures[i + 1][j + jumlah][0][0] - goodFeatures[i][j + jumlah][0][0])
                slope2 = (goodFeatures[i][j][0][1] - goodFeatures[i][j + jumlah][0][1]) / (
                        goodFeatures[i][j][0][0] - goodFeatures[i][j + jumlah][0][0])
                if slope1 < slope2:
                    print("KELUAR --", angle1)
                    directionI[j + jumlah][i] = int(3)
                else:
                    print("KELUAR ++", angle1)
                    directionI[j + jumlah][i] = int(4)
                # PEMBAGIAN EKSTRAKSI DILAKUKAN DISINI YA UNTUK BAGIAN YANG MASUK

            # PENCARIAN SISI KANAN (GOODFEATURE) DERAJAT KEMIRINGAN
            a2 = math.sqrt((goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0]) ** 2 +
                        (goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1]) ** 2)
            b2 = math.sqrt((goodFeatures[i][j + jumlah][0][0] - goodFeatures[i + 1][j][0][0]) ** 2 +
                        (goodFeatures[i][j + jumlah][0][1] - goodFeatures[i + 1][j][0][1]) ** 2)
            c2 = math.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i][j + jumlah][0][0]) ** 2 +
                        (goodFeatures[i][j][0][1] - goodFeatures[i][j + jumlah][0][1]) ** 2)
            angle2 = math.acos((c2 ** 2 + a2 ** 2 - b2 ** 2) / (2 * a2 * c2)) * 180 / math.pi
            quadrant2 = (c2 ** 2 + a2 ** 2 - b2 ** 2) / (2 * a2 * c2) * 180 / math.pi
            if quadrant2 >= -1.27222e-14:
                # MASUK
                direction[j][i] = int(1)
                slope1 = (goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1]) / (
                        goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0])
                slope2 = (goodFeatures[i][j + jumlah][0][1] - goodFeatures[i][j][0][1]) / (
                        goodFeatures[i][j + jumlah][0][0] - goodFeatures[i][j][0][0])
                if slope1 < slope2:
                    print("MASUK --", angle2)
                    directionI[j][i] = int(1)
                else:
                    print("MASUK ++", angle2)
                    directionI[j][i] = int(2)
            else:
                # KELUAR
                direction[j][i] = int(0)
                slope1 = (goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1]) / (
                        goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0])
                slope2 = (goodFeatures[i][j + jumlah][0][1] - goodFeatures[i][j][0][1]) / (
                        goodFeatures[i][j + jumlah][0][0] - goodFeatures[i][j][0][0])
                if slope1 > slope2:
                    print("KELUAR ++", angle2)
                    directionI[j][i] = int(3)
                else:
                    print("KELUAR --", angle2)
                    directionI[j][i] = int(4)


def ExtractionMethod():
    pf, nf, pm, nm = [[] for _ in range(jumlah * 2)], [[] for _ in range(jumlah * 2)], [[] for _ in range(jumlah * 2)], [[] for _ in range(jumlah * 2)]
    for j in range(jumlah * 2):
        num1, num2, num3, num4 = 0, 0, 0, 0
        for i in range(9):
            if direction[j][i] == 1:
                num1 += 1
                num3 += lengthDif[i][j]
            else:
                num2 += 1
                num4 += lengthDif[i][j]
        pf[j] = num1 / 9
        nf[j] = num2 / 9
        pm[j] = num3
        nm[j] = num4

        # MENYIMPAN FEATURE EXTRACTION METHOD I
        with open("M1F1_2AC.csv", "a") as myfile:
            for j in range((jumlah * 2) - 1):
                myfile.write(f"{str(pf[j])},{str(nf[j])},{str(pm[j])},{str(nm[j])},")
                if j == (jumlah * 2) - 2:
                    myfile.write(f"{str(pf[j + 1])},{str(nf[j + 1])},{str(pm[j + 1])},{str(nm[j + 1])}\n")

def sort_by_second(elem):
    return elem[1]

def track_visualization(images, goodFeatures):
    output_dir = 'Tracking'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize Tracking
    vect1 = [[] for _ in range(10)]
    vect2 = [[] for _ in range(10)]
    coordinate = np.zeros((50, 50, 2), dtype=np.float32)

    # Sorting data for the left and right sides
    for i in range(len(images)):
        for j in range(jumlah):
            vect1[i].append(tuple(goodFeatures[i][j][0]))
            vect2[i].append(tuple(goodFeatures[i][j + jumlah][0]))
        vect1[i] = sorted(vect1[i], key=sort_by_second)
        vect2[i] = sorted(vect2[i], key=sort_by_second)

        # Transfer sorted data to the coordinate variable
    for i in range(len(images)):
        temp1 = -1
        for j in range(jumlah - 1, -1, -1):
            temp1 += 1
            coordinate[i][temp1] = np.array(vect1[i][j])
            if j == jumlah - 1:
                for k in range(jumlah):
                    coordinate[i][k + jumlah] = np.array(vect2[i][k])

    # Draw lines and circles for visualization
    for i, image in images.items():
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for j in range(jumlah * 2 - 1):
            a = tuple(map(int, coordinate[i][j]))
            b = tuple(map(int, coordinate[i][j + 1]))
            # cv2.line(image, tuple(map(int, coordinate[i][j])), tuple(map(int, coordinate[i][j + 1])), (255, 255, 255), 2)
            cv2.line(image, a, b, (0, 255, 0), 1)

        for j in range(jumlah * 2):
            x, y = goodFeatures[i][j][0]
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 2, 8, 0)
            output_path = os.path.join(output_dir, f"tracking_{i}.png")
            cv2.imwrite(output_path, image)


if __name__ == '__main__':
    folder_path = './datasets'
    all_files = os.listdir(folder_path)

    for numVideo, rawVideo in enumerate(all_files):
        goodFeatures = [np.array([[]]) for _ in range(10)]

        GFcoordinates = {}
        valnorm = 0

        lengthDif = [[] for _ in range (9)]

        direction = np.zeros((12, 9), dtype=float)
        directionI = np.zeros((12, 9), dtype=float)
        rawImages = 0
        frames = {}
        print(rawVideo)
        frames = video2frames(rawVideo)
        rawImages = copy.copy(frames)
        print(rawImages)

        # Preprocessing
        res = median_filter(rawImages[0])
        res = high_boost_filter(rawImages[0], res, 1.5)
        res = morph(res)
        res = thresholding(res)
        #Segmentation
        res = canny(res)
        res = region_filter(res)
        #ambil 1 frame untuk menympan nilai center point
        height, width = res.shape
        X1, Y1 = (width // 2), (height // 2)
        X2, Y2 = (X1 + 22), (Y1+ 23)

        # Visualisasi titik tengah

        res = coLinear(res)

        # cv2.circle(res, (X1 ,Y1), 1, (255, 255, 255), 2, 8, 0)
        # cv2.imshow(f'nyoba', res)
        # cv2.waitKey(0)


        #Tracking
        GFcoordinates = GetGoodFeaturesPSAX(res)
        #Simpan nilai koordinat good feature
        for i in range(len(rawImages)) :
            goodFeatures[i] = goodFeatures[i].astype(np.float32)
            # for j in range(1, jumlah * 2 + 1):
            if i == 0 :
                for j in range(jumlah * 2):
                    # x = GFcoordinates[j][2][0]
                    # y = GFcoordinates[j][2][1]

                    #PSAX
                    x = GFcoordinates[j][0]
                    y = GFcoordinates[j][1]
                    goodFeatures[i] = np.append(goodFeatures[i], np.array([x, y], dtype=np.float32))
                goodFeatures[i] = goodFeatures[i].reshape((jumlah * 2, 1, 2))

        #Visualisasi Good Feature
        output_dir = 'GoodFeatures'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in rawImages.items():
            if framecount == 0:
                for i in range(jumlah*2):
                    x, y = goodFeatures[framecount][i][0]
                    output_path = os.path.join(output_dir, 'GF.png')
                    cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 2, 8, 0)
                    cv2.imwrite(output_path, image)
                break

        opticalFlowCalc(rawImages, goodFeatures)

        #Feature Extraction
        featureExtraction(goodFeatures)
        track_visualization(frames, goodFeatures)

        output_dir = 'Output'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in frames.items():
            output_path = os.path.join(output_dir, f'frame_{framecount}.png')
            cv2.imwrite(output_path, image)

        ExtractionMethod()
