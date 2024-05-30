from functools import wraps
import jwt, os, datetime, werkzeug, copy
import numpy as np
import cv2
import math

R = 65 #radius
X1, Y1 = 0, 0 #centerpoint
CCX, CCY = [0] * 100, [0] * 100

jumlah = 12
goodFeatures = [np.array([[]]) for _ in range(10)]
GFcoordinates = {}
valnorm = 0

lengthDif = [[] for _ in range (9)]

direction = np.zeros((jumlah * 2, 9), dtype=float)
directionI = np.zeros((jumlah * 2, 9), dtype=float)

jumlahFrame = 10
frames = {}
res ={}

def video2frames(video):
    rawImages = {}
    output_dir = '1.frames'
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    target_frames = jumlahFrame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_skip = max(total_frames // target_frames, 1)
    frame_count = 0
    frame_index = 0
    while frame_index < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            rawImages[frame_index] = frame
            output_image_path = os.path.join(output_dir, f'frame_{frame_index:04d}.png')
            cv2.imwrite(output_image_path, frame)
            frame_index += 1
        frame_count += 1
    cap.release()
    return rawImages

def gaussian_blur(image, kernelsize):
    output_dir = '2.gausianblur'
    os.makedirs(output_dir, exist_ok=True)
    res = np.copy(image)
    res  = cv2.GaussianBlur(image, kernelsize, 0)
    output_path = os.path.join(output_dir, 'gaussian.png')
    cv2.imwrite(output_path, res)
    return res

def median_filter(image, kernelsize):
    output_dir = '2.medianfiltered'
    os.makedirs(output_dir, exist_ok=True)
    res = np.copy(image)
    res = cv2.medianBlur(image, kernelsize)
    output_path = os.path.join(output_dir, 'median.png')
    cv2.imwrite(output_path, res)
    return res

def high_boost_filter(image, lpf, kons):
    output_dir = '3.highboost'
    os.makedirs(output_dir, exist_ok=True)
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
    output_path = os.path.join(output_dir, 'highboost.png')
    cv2.imwrite(output_path, res)
    return res


def morph(image):
    output_dir = '4.morphology'
    os.makedirs(output_dir, exist_ok=True)
    res = np.copy(image)
    # ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (3, 3))
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (1, 1))
    res = cv2.morphologyEx(image, cv2.MORPH_OPEN, ellipse)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, ellipse)
    output_path = os.path.join(output_dir, 'morphology.png')
    cv2.imwrite(output_path, res)
    return res

def thresholding(image, thr_b, thr_a):
    output_dir = '5.thresholding'
    os.makedirs(output_dir, exist_ok=True)
    res = np.copy(image)
    _, res = cv2.threshold(image, thr_b, thr_a, cv2.THRESH_BINARY) #original at 90
    output_path = os.path.join(output_dir, 'threshold.png')
    cv2.imwrite(output_path, res)
    return res

def adaptiveThresholding(image, blockSize, C, k, i):
    output_dir = '5.adaptiveThresholding'
    os.makedirs(output_dir, exist_ok=True)
    res = np.copy(image)
    res = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=blockSize, C=C)
    output_path = os.path.join(output_dir, 'adaptivethreshold.png')
    kernel = np.ones((k,k), np.uint8)
    erosion = cv2.erode(res, kernel, iterations=i)
    dilation = cv2.dilate(erosion, kernel, iterations=i)
    res = np.copy(image)
    res = dilation
    cv2.imwrite(output_path, res)
    return res


def canny(image):
    output_dir = '6.canny'
    os.makedirs(output_dir, exist_ok=True)
    res = image.copy()
    res = cv2.Canny(image, 0, 255, 3)
    output_path = os.path.join(output_dir, 'canny.png')
    cv2.imwrite(output_path, res)
    return res

def laplacian(image):
    output_dir = '6.laplacian'
    os.makedirs(output_dir, exist_ok=True)
    res = image.copy()
    res = cv2.Canny(image, 0, 255, 3)
    output_path = os.path.join(output_dir, 'canny.png')
    cv2.imwrite(output_path, res)
    return res

def region_filter(image):
    output_dir = '7.region'
    os.makedirs(output_dir, exist_ok=True)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = np.zeros_like(image)
    output_path = os.path.join(output_dir, 'region.png')
    for i in range(len(contours)):
        if len(contours[i]) > R:
            # cv2.drawContours(res, contours, i, (255, 0, 0), 1)
            cv2.drawContours(res, contours, i, (255, 0, 0), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
            cv2.imwrite(output_path, res)
    return res


def coLinear(image):
    output_dir = '8.colinear'
    os.makedirs(output_dir, exist_ok=True)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = np.zeros_like(image)
    data = [0] * 100

    idk = 0
    for i in range(len(contours)):
        if len(contours[i]) > R:
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
                    if (abs(CCX[j] - pt1[0]) < 6) and (abs(CCY[j] - pt1[1]) < 6):
                        data[j] = 0
                    else:
                        data[j] = 1
    for i in range(len(contours)):
        if data[i] == 0:
            cv2.drawContours(res, contours, i, (255, 255, 255), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))

    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    roi_contours = max(contours, key=cv2.contourArea)
    res = np.zeros_like(image)
    cv2.drawContours(res, [roi_contours], -1, (255, 255, 255), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
    output_path = os.path.join(output_dir, 'colinear.png')
    cv2.imwrite(output_path, res)
    return res


def triangleEquation(source):
    output_dir = '12.Triangle Equation'
    os.makedirs(output_dir, exist_ok=True)
    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    data1 = [[] for _ in range(200)]
    x1 = [[0] * 100 for _ in range(100)]
    y1 = [[0] * 100 for _ in range(100)]
    x2 = [[0] * 100 for _ in range(100)]
    y2 = [[0] * 100 for _ in range(100)]
    center = [0, 0]
    jum = 0
    jum1 = 0
    jum2 = 0
    noCon = []
    idk = 0
    for i in range(len(contours)):

        if len(contours[i]) > R:
            x, y, w, h = cv2.boundingRect(contours[i])
            center_x = x + w // 2
            center_y = y + h // 2
            pt = (center_x, center_y)
            # pt = (X1, Y1)
            #check titik tengah berada di dalam kontur ROI atau tidak
            out = cv2.pointPolygonTest(contours[i], pt, False)
            print(out)
            if out > 0:
                jum1 += 1
            else:
                noCon.append(i)
                jum2 += 1
        idk += 1

    if jum1 > 0:
        print("bentuk=1")
        res = np.zeros_like(source)
        for m in range(len(contours)):
            if len(contours[m]) > R:
                cv2.drawContours(res, contours, m, (255, 0, 0), 1, lineType=8,)
        output_path = os.path.join(output_dir, 'TriangleEquation.png')
        cv2.imwrite(output_path, res)
        return res

    if jum2 == 1:
        print("bentuk=2")
        j = 0
        res = np.zeros_like(source)
        for m in range(len(contours)):
            if len(contours[m]) > R:
                k = 0
                for i in range (len(contours[m]) - 7):
                    p1 = contours[m][i][0]
                    p2 = contours[m][i + 1][0]
                    p3 = contours[m][i + 2][0]
                    p4 = contours[m][i + 3][0]
                    p5 = contours[m][i + 4][0]
                    p6 = contours[m][i + 5][0]
                    p7 = contours[m][i + 6][0]
                    d = int(np.sqrt(pow((p1[0] - p7[0]), 2.0) + pow((p1[1] - p7[1]), 2.0))) + \
                        int(np.sqrt(pow((p2[0] - p6[0]), 2.0) + pow((p2[1] - p6[1]), 2.0))) + \
                        int(np.sqrt(pow((p3[0] - p5[0]), 2.0) + pow((p3[1] - p5[1]), 2.0)))

                    if d <= 15:
                        data1[k] = i + 3
                        k += 1
                        CCX[j] = p4[0]
                        CCY[j] = p4[1]
                        cv2.line(source, (int(CCX[j] - 1), int(CCY[j])), (int(CCX[j] + 1), int(CCY[j])), (255, 0, 0), thickness=1)
                        cv2.line(source, (int(CCX[j]), int(CCY[j] - 1)), (int(CCX[j]), int(CCY[j] + 1)), (255, 0, 0), thickness=1)

                print(k)
                k = 0
                for i in range (len(contours[m]) - 7):
                    p1 = contours[m][i][0]
                    p2 = contours[m][i + 1][0]
                    p3 = contours[m][i + 2][0]
                    p4 = contours[m][i + 3][0]
                    p5 = contours[m][i + 4][0]
                    p6 = contours[m][i + 5][0]
                    p7 = contours[m][i + 6][0]
                    d = int(np.sqrt(pow((p1[0] - p7[0]), 2.0) + pow((p1[1] - p7[1]), 2.0))) + \
                        int(np.sqrt(pow((p2[0] - p6[0]), 2.0) + pow((p2[1] - p6[1]), 2.0))) + \
                        int(np.sqrt(pow((p3[0] - p5[0]), 2.0) + pow((p3[1] - p5[1]), 2.0)))
                    if d <= 3 :
                        data1[k] = i + 3
                        k += 1
                        CCX[j] = p4[0]
                        CCY[j] = p4[1]
                        cv2.line(source, (int(CCX[j] - 1), int(CCY[j])), (int(CCX[j] + 1), int(CCY[j])), (255, 255, 255), thickness=1)
                        cv2.line(source, (int(CCX[j]), int(CCY[j] - 1)), (int(CCX[j]), int(CCY[j] + 1)), (255, 255, 255), thickness=1)

                center[0] = X1
                center[1] = Y1
                print(center)
                jum = 0
                min = 2000.0
                p1 = contours[m][data1[0]][0]
                for i in range(data1[1], data1[1] + R):
                    p = contours[m][i][0]
                    a1 = np.sqrt(pow((p1[0] - p[0]), 2.0) + pow((p1[1] - p[1]), 2.0))
                    b1 = np.sqrt(pow((center[0] - p[0]), 2.0) + pow((center[1] - p[1]), 2.0))
                    c1 = np.sqrt(pow((center[0] - p1[0]), 2.0) + pow((center[1] - p1[1]), 2.0))
                    alpha = math.acos(((b1 * b1) + (c1 * c1) - (a1 * a1))/ (2 * b1 * c1)) * (180/math.pi)

                    if (alpha < min):
                        min = alpha
                        jum = i

                jum1 = 0
                min = 2000.0
                p1 = contours[m][jum][0]
                for i in range(data1[0], data1[0] + R):
                    p = contours[m][i][0]
                    a1 = np.sqrt(pow((p1[0] - p[0]), 2.0) + pow((p1[1] - p[1]), 2.0))
                    b1 = np.sqrt(pow((center[0] - p[0]), 2.0) + pow((center[1] - p[1]), 2.0))
                    c1 = np.sqrt(pow((center[0] - p1[0]), 2.0) + pow((center[1] - p1[1]), 2.0))
                    alpha = math.acos(((b1 * b1) + (c1 * c1) - (a1 * a1))/ (2 * b1 * c1)) * (180/math.pi)
                    if (alpha < min):
                        min = alpha
                        jum1 = i

                p1 = contours[m][jum][0]
                p2 = contours[m][jum1][0]
                x1[0][0] = p1[0]
                y1[0][0] = p1[1]
                x2[0][0] = p2[0]
                y2[0][0] = p2[1]

                cv2.circle(source, (int(p1[0]), int(p1[1])), 1, (255, 255, 255), 2, 8, 0)
                cv2.circle(source, (int(p2[0]), int(p2[1])), 1, (255, 255, 255), 2, 8, 0)
                # cv2.line(source, (int(x1[0][0]), int(y1[0][0])), (int(x2[0][0]), int(y2[0][0])), (255, 0, 0), thickness=4)

            j += 1

        # contours, hierarchy = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # res = np.zeros_like(source)


        for m in range(len(contours)):

            # cv2.drawContours(res, contours, m, (255, 0, 0), 1, lineType=8,)
            if len(contours[m]) > R:
                end_idx = 0

                # find endpoint index
                # print(len(contours[m]))
                for i in range(len(contours[m])):
                    checkpoint = [contours[m][i][0][0], contours[m][i][0][1]]
                    endpoint = [p2[0], p2[1]]
                    result_variable = np.allclose(np.array(checkpoint), np.array(endpoint), atol =1) #atol nilai toleransi mendekati
                    if (result_variable == True):
                        end_idx = i
                        break

                #redraw contour from startpoint to endpoint
                for i in range(len(contours[m])):
                    checkpoint = [contours[m][i][0][0], contours[m][i][0][1]]
                    startpoint = [p1[0], p1[1]]
                    result_variable = np.allclose(np.array(checkpoint), np.array(startpoint), atol =1)
                    if (result_variable == True):
                        contour_part = [contours[m][i:end_idx+1]]
                        cv2.drawContours(res, contour_part, -1,(255, 0, 0), 1, lineType=8,)
                        break

        output_path = os.path.join(output_dir, 'TriangleEquation.png')
        cv2.imwrite(output_path, res)
        return res
    if jum2 == 2:
        print("bentuk=3")
    if jum2 == 3:
        print("bentuk=4")
    if jum2 == 4 or jum2 == 5:
        print("bentuk=5")

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

def GetGoodFeaturesPSAX(res):
    temp1, temp2, = 0, 0
    count = 0
    banyak = jumlah * 2
    garis = np.zeros(res.shape, dtype=res.dtype)
    hasil = np.zeros(res.shape, dtype=res.dtype)
    color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    minRect = [cv2.minAreaRect(contour) for contour in contours]

    for i in range(len(contours)):
        cv2.drawContours(garis, contours, i, color)
        rect_points = cv2.boxPoints(minRect[i])

    kondisi1 = rect_points[2][0] - rect_points[1][0]
    kondisi2 = rect_points[1][1] - rect_points[2][1]

    print("kondisi 1 :" + str(kondisi1))
    print("kondisi 2 :" + str(kondisi2))

    global valnorm
    if kondisi1 < kondisi2:
        print('kanan')
        valnorm = math.sqrt(pow((rect_points[3][0] - rect_points[2][0], 2)) + pow(rect_points[3][1] - rect_points[2][1], 2))
    else:
        print('kiri')
        # valnorm = math.sqrt(pow(rect_points[2][0] - rect_points[3][0], 2) + pow(rect_points[2][1] - rect_points[3][1], 2))
        valnorm = math.sqrt(pow((rect_points[3][0] - rect_points[2][0], 2)) + pow(rect_points[3][1] - rect_points[2][1], 2))

    coordinate1 = []  # Create an empty list for storing coordinates

    for i in range(len(contours)):
        for j in range(len(contours[i]) // 2): #kontur yang terhubung memiliki len 2 sehingga perlu dibagi 2  terlebih dahlu
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

    goodFeatures = []
    for i in range(1, banyak + 1):
        goodFeatures.append(coordinate2[i])

    return goodFeatures


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

    for i in range(len(sources)):
        sources[i] = cv2.cvtColor(sources[i], cv2.COLOR_BGR2GRAY)
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
                    if (j > 0 and j < jumlah - 1) or (j > jumlah and j < (jumlah * 2) - 1):
                        # jarak titik yang dihitung
                        length[1] = np.sqrt((goodFeatures[i][j - 1][0][0] - goodFeatures[i + 1][j - 1][0][0]) ** 2 + (goodFeatures[i][j - 1][0][1] - goodFeatures[i + 1][j - 1][0][1]) ** 2) / valnorm * 100
                        # jarak titik tetangga (+1)
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
                    if j == 0 or j == jumlah:
                        length[3] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2)
                        angleNorm = findAngle(goodFeatures[i][j + 1][0][0], goodFeatures[i][j + 1][0][1], goodFeatures[i + 1][j + 1][0][0], goodFeatures[i + 1][j + 1][0][1])
                        s = np.sin(angleNorm * np.pi / 180)
                        c = np.cos(angleNorm * np.pi / 180)
                        P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                        goodFeatures[i][j][0] = P

                    elif j == (jumlah - 1) or j == (jumlah * 2) - 1:
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

def findCenterPoint(source):
    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if len(contours[i]) > R:
            x, y, w, h = cv2.boundingRect(contours[i])
            center_x = x + w // 2
            center_y = y + h // 2
            break

    return center_x, center_y

def featureExtractionPSAX(goodFeatures):
    for j in range(jumlah):
        for i in range(9):
            #vektor AC
            vec_AC = (X1 - goodFeatures[i][j][0][0], Y1 - goodFeatures[i][j][0][1])
            #vektor AB
            vec_AB = (goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0], goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1])
            #dot product dari vektor AC dan AB
            dot_product = vec_AC[0] * vec_AB[0] + vec_AC[1] * vec_AB[1]
            #magnitudo dari vektor AC dan AB
            mag_AC = math.sqrt(vec_AC[0]**2 + vec_AC[1]**2)
            mag_AB = math.sqrt(vec_AB[0]**2 + vec_AB[1]**2)
            #kosinus sudut
            cos_angle = dot_product / (mag_AC * mag_AB)
            #sudut dalam radian
            angle_rad = math.acos(cos_angle)
            # Konversi sudut dari radian ke derajat
            angle_deg = math.degrees(angle_rad)
            if angle_deg > 90:
                print("Keluar")
                direction[j][i] = int(0)
            else:
                print("Masuk")
                direction[j][i] = int(1)


def ExtractionMethodI():
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

    with open("M1F1_PSAX.csv", "a") as myfile:
        for j in range((jumlah * 2) - 1):
            myfile.write(f"{str(pf[j])},{str(nf[j])},{str(pm[j])},{str(nm[j])},")
            if j == (jumlah * 2) - 2:
                myfile.write(f"{str(pf[j + 1])},{str(nf[j + 1])},{str(pm[j + 1])},{str(nm[j + 1])}\n")

def ExtractionMethodII():
    pf, nf= [[] for _ in range(jumlah * 2)], [[] for _ in range(jumlah * 2)]
    for j in range(jumlah * 2):
        num1, num2 = 0, 0, 0, 0
        for i in range(9):
            if direction[j][i] == 1:
                num1 += 1
            else:
                num2 += 1

        pf[j] = num1 / 9
        nf[j] = num2 / 9

    with open("M1F2_PSAX.csv", "a") as myfile:
        for j in range((jumlah * 2) - 1):
            myfile.write(f"{str(pf[j])},{str(nf[j])},")
            if j == (jumlah * 2) - 2:
                myfile.write(f"{str(pf[j + 1])},{str(nf[j + 1])}\n")


def sort_by_second(elem):
    return elem[1]

def goodFeaturesVisualization(rawImages):
    output_dir = '9.GoodFeatures'
    os.makedirs(output_dir, exist_ok=True)
    for framecount, image in rawImages.items():
        if framecount == 0:
            for i in range(jumlah*2):
                x, y = goodFeatures[framecount][i][0]
                output_path = os.path.join(output_dir, 'GF.png')
                cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 2, 8, 0)
                cv2.imwrite(output_path, image)
            break

def track_visualization(images, goodFeatures):
    output_dir = '10.Tracking'
    os.makedirs(output_dir, exist_ok=True)
    trackingresult = {}

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
            trackingresult[i] = image
        output_path = os.path.join(output_dir, f"tracking_{i}.png")
        cv2.imwrite(output_path, image)

    return trackingresult

def track_visualization2(images, goodFeatures):
    for i in range(len(images)-1):
        for j in range(jumlah * 2):
            output_path = os.path.join('10.Tracking', f'TrackingLK.png')
            gfx_awal = int(goodFeatures[0][j][0][0])
            gfy_awal = int(goodFeatures[0][j][0][1])
            gfx_akhir = int(goodFeatures[len(images) - 1][j][0][0])
            gfy_akhir = int(goodFeatures[len(images) - 1][j][0][1])
            cv2.circle(images[0], (gfx_awal, gfy_awal), 1, (255, 255, 255), 2, 8, 0)
            cv2.line(images[0], (gfx_awal, gfy_awal), (gfx_akhir, gfy_akhir), (255, 255, 255), 1)
            cv2.imwrite(output_path, images[0])

def frames2video(images):
    img_array = []
    for i, img in images.items():
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':
    videofile = "normalc_5.avi"
    rawVideo = "./DatasetsPSAX/"+ videofile
    print("\nReceived image File name : " + videofile)
    frames = video2frames(rawVideo)
    print('total frames :'+str(len(frames)))
    rawImages = copy.deepcopy(frames)

    #Preprocessing
    flow_choice = input("Enter 1 to run Flow 1 or 2 to run Flow 2: ")
    flow_choice = int(flow_choice)

    if flow_choice == 1:
        res = gaussian_blur(rawImages[0], (5,5))
        res = high_boost_filter(rawImages[0], res, 2.5)
        res = morph(res)
        res = adaptiveThresholding(res, 3, 1, 3, 2) #(blockSize=3, C=1, kernel=3, iterations=2)
    elif flow_choice == 2:
        res = median_filter(rawImages[0], 21)
        res = high_boost_filter(rawImages[0], res, 2.5)
        res = morph(res)
        res = thresholding(res, 10, 255)
    else:
        print("Invalid choice. Please enter either 1 or 2.")

    #Segmentation
    res = canny(res)
    res = region_filter(res)

    height, width = res.shape
    X1, Y1 = (width // 2), (height // 2) #ambil 1 frame untuk menympan nilai center point

    res = coLinear(res)
    res = triangleEquation(res)
    cv2.imshow('Triangle', res)
    cv2.waitKey(0)


    X1, Y1 = findCenterPoint(res)
    # cv2.circle(res, (X1, Y1), 5, (255, 0, 0), -1)
    # cv2.imshow('Titik Tengah', res)

    GFcoordinates = GetGoodFeaturesPSAX(res)

    #Simpan nilai koordinat good feature
    for i in range(len(rawImages)) :
        goodFeatures[i] = goodFeatures[i].astype(np.float32)
        if i == 0:
            #Without intersect
            for j in range(jumlah * 2):
                x = GFcoordinates[j][0]
                y = GFcoordinates[j][1]
                goodFeatures[i] = np.append(goodFeatures[i], np.array([x, y], dtype=np.float32))
            goodFeatures[i] = goodFeatures[i].reshape((jumlah * 2, 1, 2))

    # Visualisasi Good Feature
    goodFeaturesVisualization(rawImages)

    #Tracking
    opticalFlowCalcwithNormalization(rawImages, goodFeatures)

    #Visualisasi Tracking
    visualFrames1 = copy.deepcopy(frames)
    res = track_visualization(visualFrames1, goodFeatures)
    visualFrames2 = copy.deepcopy(frames)
    track_visualization2(visualFrames2, goodFeatures)

    #Feature Extraction
    featureExtractionPSAX(goodFeatures)
    ExtractionMethodI() # 24 x 4 Fitur (Arah +, Arah -, Jarak +, Jarak -)
    ExtractionMethodII() # 24 x 2 Fitur (Arah +, Arah -)

    res = frames2video(res)