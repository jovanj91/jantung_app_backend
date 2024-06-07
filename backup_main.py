from flask import Flask , request, make_response, jsonify, send_file
from flask_restful import Resource, Api
from flask_cors import CORS


from flask_security import Security, current_user, SQLAlchemySessionUserDatastore, permissions_accepted, roles_required, auth_token_required
from flask_security.utils import verify_password, hash_password, login_user
from config import DevelopmentConfig
# from google.cloud import storage

from functools import wraps
from database import db_session, init_db
from models import User, Role, RolesUsers, PatientData, HeartCheck
import jwt, os, datetime, werkzeug, copy, io
import joblib
from sklearn import preprocessing
import pickle
import pandas as pd
import numpy as np
import cv2
import math

app = Flask(__name__)
api = Api(app)

app.config.from_object(DevelopmentConfig)

app.teardown_appcontext(lambda exc: db_session.close())

bucket_name = "jantungappbackend.appspot.com"
user_datastore = SQLAlchemySessionUserDatastore(db_session, User, Role)
security = Security(app, user_datastore)


# storage_client = storage.Client()


class HelloWorld(Resource):
    @auth_token_required
    def get(self):
        return f"<p>Hello, World! {current_user.username}</p>"


class RegisterUser(Resource):
    def post(self):
        usernameInput = request.json['username']
        useremailInput = request.json['email']
        passwordInput = request.json['password'].encode('utf-8')

        user = security.datastore.find_user(email= useremailInput)
        if not user:
            security.datastore.create_user(email=useremailInput, password=hash_password(passwordInput), username=usernameInput, roles=["user"])
            db_session.commit()
            try:
                db_session.close()
                return make_response(jsonify(message="Registration successful"), 201)
            except Exception as e:
                db_session.rollback()
                return make_response(jsonify(error="Registration failed", details=str(e)), )
        else:
            return make_response(jsonify(error="Email already registered" ), 409  )

class UploadVideo(Resource):
    # @token_requried
    def post(self):
        videofile = request.files['video']
        filename = werkzeug.utils.secure_filename(videofile.filename)
        print("\nReceived image File name : " + videofile.filename)
        videofile.save("./uploadedvideo/" + filename)
        return jsonify({
            "message" : "file uploaded successfully"
        })

class InputPatientData(Resource):
    @auth_token_required
    @roles_required('user')
    def post(self):
        nameInput = request.json['name']
        genderInput = request.json['gender']
        dobInput = request.json['dob']
        uname = db_session.query(User).filter_by(username=current_user.username).first()
        # bucket = storage_client.bucket(bucket_name)
        # user_directory = f'{current_user.username}_data/'
        # # blobs = bucket.list_blobs(prefix=user_directory + '/')
        # # if blobs:
        # patient_directory = f'{nameInput}_data'
        # blob = bucket.blob(user_directory + patient_directory + '/')
        # blob.upload_from_string('')

        dob_date = datetime.datetime.strptime(str(dobInput), "%Y-%m-%d")
        current_date = datetime.datetime.now()
        age = current_date.year - dob_date.year - ((current_date.month, current_date.day) < (dob_date.month, dob_date.day))

        inputData = PatientData(patient_name =nameInput, dob=dobInput, gender=genderInput, age=age, user=uname)

        db_session.add(inputData)
        db_session.commit()
        try:
            db_session.close()
            return make_response(jsonify(message="Data added successfully"), 201)
        except Exception as e:
            db_session.rollback()
            return make_response(jsonify(error="Data failed to be added", details=str(e)), 409)

class GetPatientsData(Resource):
    @auth_token_required
    @roles_required('user')
    def get(self):
        patients = db_session.query(PatientData).filter(PatientData.user_id == current_user.id)
        patientList = []
        for patient in patients:
            heartCheck = db_session.query(HeartCheck).filter(HeartCheck.patient_id == patient.id).order_by(HeartCheck.checked_at.desc()).first()
            if heartCheck != None :
                lastCheck = heartCheck.checkResult
            else :
                lastCheck = "Not Checked Yet"
            patientList.append({
                'patientId' : patient.id,
                'patientName' : patient.patient_name,
                'patientAge' : patient.age,
                'patientGender' : patient.gender,
                'patientDob' : patient.dob,
                'lastCheck' : lastCheck
            })
        return make_response(jsonify({
            'data' : patientList
        }), 201)


class GetPatientCheckHistory(Resource):
    @auth_token_required
    @roles_required('user')
    def get(self):
        patient_id = request.args.get('patient_id')
        histories =  db_session.query(HeartCheck, PatientData).join(PatientData, HeartCheck.patient_id == PatientData.id).filter(HeartCheck.patient_id == patient_id)
        historyList = []
        if histories:
            for heart_check, patient_data in histories:
                historyList.append({
                    'checkResult' : heart_check.checkResult,
                    'checkedAt' : heart_check.checked_at,
                })
        else:
            historyList.append("No History Data")
        return make_response(jsonify({
            'data' : historyList
        }), 201)


#Preprocessing, Segmentation, GoodFeature, Tracking and Feature Extraction
class Preprocessing(Resource):
    def video2frames(self, video):
        rawImages = {}
        output_dir = '1.frames'
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video)
        target_frames = self.jumlahFrame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_skip = max(total_frames // target_frames, 1)
        print('frameskip : ' + str(frame_skip))
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

    def median_filter(self, image):
        output_dir = '2.medianfiltered'
        os.makedirs(output_dir, exist_ok=True)
        res = np.copy(image)
        kernelsize = 27 #5 edge more complete
        res = cv2.medianBlur(image, kernelsize)
        output_path = os.path.join(output_dir, 'median.png')
        cv2.imwrite(output_path, res)
        return res

    def high_boost_filter(self, image, lpf, kons):
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


    def morph(self, image):
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

    def thresholding(self, image):
        output_dir = '5.thresholding'
        os.makedirs(output_dir, exist_ok=True)
        res = np.copy(image)
        _, res = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY) #original at 90
        output_path = os.path.join(output_dir, 'threshold.png')
        cv2.imwrite(output_path, res)
        return res

    def canny(self, image):
        output_dir = '6.canny'
        os.makedirs(output_dir, exist_ok=True)
        res = image.copy()
        res = cv2.Canny(image, 0, 255, 3)
        output_path = os.path.join(output_dir, 'canny.png')
        cv2.imwrite(output_path, res)
        return res

    def region_filter(self, image):
        output_dir = '7.region'
        os.makedirs(output_dir, exist_ok=True)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res = np.zeros_like(image)
        output_path = os.path.join(output_dir, 'region.png')
        for i in range(len(contours)):
            if len(contours[i]) > self.R:
                # cv2.drawContours(res, contours, i, (255, 0, 0), 1)
                cv2.drawContours(res, contours, i, (255, 0, 0), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
                cv2.imwrite(output_path, res)
        return res


    def coLinear(self, image):
        output_dir = '8.colinear'
        os.makedirs(output_dir, exist_ok=True)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        res = np.zeros_like(image)
        data = [0] * 100

        idk = 0
        for i in range(len(contours)):
            if len(contours[i]) > self.R:
                pt = contours[i][len(contours[i]) // 4][0]
                self.CCX[idk] = pt[0]
                self.CCY[idk] = pt[1]
                data[idk] = 0
            else:
                self.CCX[idk] = 0
                self.CCY[idk] = 0
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
                    out = self.intersectionLine(self.X1, self.Y1, self.CCX[j], self.CCY[j], pt1[0], pt1[1], pt2[0], pt2[1])
                    if out == 1:
                        if (abs(self.CCX[j] - pt1[0]) < 6) and (abs(self.CCY[j] - pt1[1]) < 6):
                            data[j] = 0
                        else:
                            data[j] = 1
        for i in range(len(contours)):
            if data[i] == 0:
                cv2.drawContours(res, contours, i, (255, 255, 255), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
                # output_path = os.path.join(output_dir, 'colinear.png')
                # cv2.imwrite(output_path, res)

        contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        roi_contours = max(contours, key=cv2.contourArea)
        res = np.zeros_like(image)
        cv2.drawContours(res, [roi_contours], -1, (255, 255, 255), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
        output_path = os.path.join(output_dir, 'colinear.png')
        cv2.imwrite(output_path, res)
        return res


    def triangleEquation(self, source):
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
            if len(contours[i]) > self.R:
                pt = (self.X1, self.Y1)
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
                if len(contours[m]) > self.R:
                    cv2.drawContours(res, contours, m, (255, 0, 0), 1, lineType=8,)
            output_path = os.path.join(output_dir, 'TriangleEquation.png')
            cv2.imwrite(output_path, res)
            return res

        if jum2 == 1:
            print("bentuk=2")
            j = 0
            res = np.zeros_like(source)
            for m in range(len(contours)):
                if len(contours[m]) > self.R:
                    k = 0
                    for i in range (len(contours[m]) - 7):
                        print(f'i : {i}')
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
                        # print(f'd{i} =' + str(d))
                        if d <= 15:
                            data1[k] = i + 3
                            k += 1
                            self.CCX[j] = p4[0]
                            self.CCY[j] = p4[1]
                            cv2.line(source, (int(self.CCX[j] - 1), int(self.CCY[j])), (int(self.CCX[j] + 1), int(self.CCY[j])), (255, 0, 0), thickness=1)
                            cv2.line(source, (int(self.CCX[j]), int(self.CCY[j] - 1)), (int(self.CCX[j]), int(self.CCY[j] + 1)), (255, 0, 0), thickness=1)

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
                            self.CCX[j] = p4[0]
                            self.CCY[j] = p4[1]
                            cv2.line(source, (int(self.CCX[j] - 1), int(self.CCY[j])), (int(self.CCX[j] + 1), int(self.CCY[j])), (255, 255, 255), thickness=1)
                            cv2.line(source, (int(self.CCX[j]), int(self.CCY[j] - 1)), (int(self.CCX[j]), int(self.CCY[j] + 1)), (255, 255, 255), thickness=1)

                    center[0] = self.X1
                    center[1] = self.Y1
                    print(center)
                    jum = 0
                    min = 2000.0
                    p1 = contours[m][data1[0]][0]
                    for i in range(data1[1], data1[1] + self.R):
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
                    for i in range(data1[0], data1[0] + self.R):
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
                    # cv2.line(source, (int(x1[0][0]), int(y1[0][0])), (int(x2[0][0]), int(y2[0][0])), (255, 0, 0), thickness=4)

                j += 1

            # contours, hierarchy = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for m in range(len(contours)):
                if len(contours[m]) > self.R:
                    # cv2.drawContours(res, contours, m, (255, 0, 0), 1, lineType=8,)
                    end_idx = 0

                    # find endpoint index
                    print(len(contours[m]))
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
                        endpoint = [p1[0], p1[1]]
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

    def intersectionLine(self, x1, y1, x2, y2, x3, y3, x4, y4):
        m1, c1 = self.straightLine(x1, y1, x2, y2)
        m2, c2 = self.straightLine(x3, y3, x4, y4)

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

    def straightLine(self, x1, y1, x2, y2):
        x = x1 - x2
        if x == 0:
            m = 1e6
        else:
            m = (y1 - y2) / x
        b = y1 - m * x1
        return m, b

    def GetGoodFeaturesPSAX(self, res):
        temp1, temp2, = 0, 0
        count = 0
        banyak = self.jumlah * 2
        garis = np.zeros(res.shape, dtype=res.dtype)
        hasil = np.zeros(res.shape, dtype=res.dtype)
        color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))

        contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        minRect = [cv2.minAreaRect(contour) for contour in contours]

        for i in range(len(contours)):
            cv2.drawContours(garis, contours, i, color)
            rect_points = cv2.boxPoints(minRect[i])

        # kondisi1 = rect_points[3][0] - rect_points[0][0]
        # kondisi2 = rect_points[0][1] - rect_points[3][1]

        kondisi1 = rect_points[2][0] - rect_points[1][0]
        kondisi2 = rect_points[1][1] - rect_points[2][1]


        if kondisi1 < kondisi2:
            self.valnorm = math.sqrt(pow((rect_points[3][0] - rect_points[2][0], 2)) + pow(rect_points[3][1] - rect_points[2][1], 2))
        else:

            # valnorm = math.sqrt(pow(rect_points[2][0] - rect_points[3][0], 2) + pow(rect_points[2][1] - rect_points[3][1], 2))
            self.valnorm = math.sqrt(pow(rect_points[3][0] - rect_points[2][0], 2) + pow(rect_points[3][1] - rect_points[2][1], 2))

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


    def findAngle(self, x1, y1, x2, y2):
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

    def opticalFlowCalcwithNormalization(self, sources, goodFeatures):
        thresh_diff = 20.0
        termCrit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)
        winSize = (21, 21)
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
                    length[0] = np.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j][0][0]) ** 2 + (goodFeatures[i][j][0][1] - goodFeatures[i + 1][j][0][1]) ** 2) / self.valnorm * 100
                    if length[0] > thresh_diff:
                        if (j > 0 and j < self.jumlah - 1) or (j > self.jumlah and j < (self.jumlah * 2) - 1):
                            length[1] = np.sqrt((goodFeatures[i][j - 1][0][0] - goodFeatures[i + 1][j - 1][0][0]) ** 2 + (goodFeatures[i][j - 1][0][1] - goodFeatures[i + 1][j - 1][0][1]) ** 2) / self.valnorm * 100
                            length[2] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2) / self.valnorm * 100

                            if length[1] < thresh_diff:
                                length[3] = np.sqrt((goodFeatures[i][j - 1][0][0] - goodFeatures[i + 1][j - 1][0][0]) ** 2 + (goodFeatures[i][j - 1][0][1] - goodFeatures[i + 1][j - 1][0][1]) ** 2)
                                angleNorm = self.findAngle(goodFeatures[i][j - 1][0][0], goodFeatures[i][j - 1][0][1], goodFeatures[i + 1][j - 1][0][0], goodFeatures[i + 1][j - 1][0][1])
                                s = np.sin(angleNorm * np.pi / 180)
                                c = np.cos(angleNorm * np.pi / 180)
                                P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                                goodFeatures[i + 1][j][0] = P

                            elif length[2] < thresh_diff:
                                length[3] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2)
                                angleNorm = self.findAngle(goodFeatures[i][j + 1][0][0], goodFeatures[i][j + 1][0][1], goodFeatures[i + 1][j + 1][0][0], goodFeatures[i + 1][j + 1][0][1])
                                s = np.sin(angleNorm * np.pi / 180)
                                c = np.cos(angleNorm * np.pi / 180)
                                P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                                goodFeatures[i + 1][j][0] = P

                for j in range(len(goodFeatures[i])):
                    length[0] = np.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j][0][0]) ** 2 + (goodFeatures[i][j][0][1] - goodFeatures[i + 1][j][0][1]) ** 2) / self.valnorm * 100
                    if length[0] > thresh_diff:
                        if j == 0 or j == self.jumlah:
                            length[3] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2)
                            angleNorm = self.findAngle(goodFeatures[i][j + 1][0][0], goodFeatures[i][j + 1][0][1], goodFeatures[i + 1][j + 1][0][0], goodFeatures[i + 1][j + 1][0][1])
                            s = np.sin(angleNorm * np.pi / 180)
                            c = np.cos(angleNorm * np.pi / 180)
                            P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                            goodFeatures[i][j][0] = P

                        elif j == (self.jumlah + 1) or j == (self.jumlah * 2):
                            length[3] = np.sqrt((goodFeatures[i][j - 1][0][0] - goodFeatures[i + 1][j - 1][0][0]) ** 2 + (goodFeatures[i][j - 1][0][1] - goodFeatures[i + 1][j - 1][0][1]) ** 2)
                            angleNorm = self.findAngle(goodFeatures[i][j - 1][0][0], goodFeatures[i][j - 1][0][1], goodFeatures[i][j - 1][0][0], goodFeatures[i][j - 1][0][1])
                            s = np.sin(angleNorm * np.pi / 180)
                            c = np.cos(angleNorm * np.pi / 180)
                            P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                            goodFeatures[i][j][0] = P

        for i in range(9):
            for j in range(len(goodFeatures[i])):
                length = (math.sqrt(((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j][0][0]) ** 2) + ((goodFeatures[i][j][0][1] - goodFeatures[i + 1][j][0][1]) ** 2)) / self.valnorm) * 100
                self.lengthDif[i].append(length)

    def findCenterPoint(self, source):
        contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if len(contours[i]) > self.R:
                x, y, w, h = cv2.boundingRect(contours[i])
                center_x = x + w // 2
                center_y = y + h // 2
                break

        return center_x, center_y

    def featureExtractionPSAX(self, goodFeatures):
        for j in range(self.jumlah):
            for i in range(9):
                vec_AC = (self.X1 - goodFeatures[i][j][0][0], self.Y1 - goodFeatures[i][j][0][1])
                vec_AB = (goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0], goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1])
                dot_product = vec_AC[0] * vec_AB[0] + vec_AC[1] * vec_AB[1]
                mag_AC = math.sqrt(vec_AC[0]**2 + vec_AC[1]**2)
                mag_AB = math.sqrt(vec_AB[0]**2 + vec_AB[1]**2)
                cos_angle = dot_product / (mag_AC * mag_AB)
                angle_rad = math.acos(cos_angle)
                angle_deg = math.degrees(angle_rad)
                if angle_deg > 90:
                    # print("Keluar")
                    self.direction[j][i] = int(0)
                else:
                    # print("Masuk")
                    self.direction[j][i] = int(1)


    def ExtractionMethod(self):
        pf, nf= [[] for _ in range(self.jumlah * 2)], [[] for _ in range(self.jumlah * 2)]
        for j in range(self.jumlah * 2):
            num1, num2 = 0, 0
            for i in range(9):
                if self.direction[j][i] == 1:
                    num1 += 1
                else:
                    num2 += 1

            pf[j] = num1 / 9
            nf[j] = num2 / 9

        # MENYIMPAN FEATURE EXTRACTION METHOD I
        with open("M1F2_PSAX.csv", "a") as myfile:
            for j in range((self.jumlah * 2) - 1):
                myfile.write(f"{str(pf[j])},{str(nf[j])},")
                if j == (self.jumlah * 2) - 2:
                    myfile.write(f"{str(pf[j + 1])},{str(nf[j + 1])}\n")

    def sort_by_second(self, elem):
        return elem[1]

    def goodFeaturesVisualization(self, rawImages):
        output_dir = '9.GoodFeatures'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in rawImages.items():
            if framecount == 0:
                for i in range(self.jumlah*2):
                    x, y = self.goodFeatures[framecount][i][0]
                    output_path = os.path.join(output_dir, 'GF.png')
                    cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 2, 8, 0)
                    cv2.imwrite(output_path, image)
                break

    def track_visualization(self, images, goodFeatures):
        output_dir = '10.Tracking'
        os.makedirs(output_dir, exist_ok=True)
        trackingresult = {}
        # Visualize Tracking
        vect1 = [[] for _ in range(10)]
        vect2 = [[] for _ in range(10)]
        coordinate = np.zeros((50, 50, 2), dtype=np.float32)

        # Sorting data for the left and right sides
        for i in range(len(images)):
            for j in range(self.jumlah):
                vect1[i].append(tuple(goodFeatures[i][j][0]))
                vect2[i].append(tuple(goodFeatures[i][j + self.jumlah][0]))
            vect1[i] = sorted(vect1[i], key=self.sort_by_second)
            vect2[i] = sorted(vect2[i], key=self.sort_by_second)

        # Transfer sorted data to the coordinate variable
        for i in range(len(images)):
            temp1 = -1
            for j in range(self.jumlah - 1, -1, -1):
                temp1 += 1
                coordinate[i][temp1] = np.array(vect1[i][j])
                if j == self.jumlah - 1:
                    for k in range(self.jumlah):
                        coordinate[i][k + self.jumlah] = np.array(vect2[i][k])


        # Draw lines and circles for visualization
        for i, image in images.items():
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for j in range(self.jumlah * 2 - 1):
                a = tuple(map(int, coordinate[i][j]))
                b = tuple(map(int, coordinate[i][j + 1]))
                # cv2.line(image, tuple(map(int, coordinate[i][j])), tuple(map(int, coordinate[i][j + 1])), (255, 255, 255), 2)
                cv2.line(image, a, b, (0, 255, 0), 1)

            for j in range(self.jumlah * 2):
                x, y = goodFeatures[i][j][0]
                cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 2, 8, 0)
                trackingresult[i] = image
            output_path = os.path.join(output_dir, f"tracking_{i}.png")
            cv2.imwrite(output_path, image)

        return trackingresult

    def track_visualization2(self, images, goodFeatures):
        for i in range(len(images)-1):
            for j in range(self.jumlah * 2):
                output_path = os.path.join('10.Tracking', f'TrackingLK.png')
                gfx_awal = int(goodFeatures[0][j][0][0])
                gfy_awal = int(goodFeatures[0][j][0][1])
                gfx_akhir = int(goodFeatures[len(images) - 1][j][0][0])
                gfy_akhir = int(goodFeatures[len(images) - 1][j][0][1])
                cv2.circle(images[0], (gfx_awal, gfy_awal), 1, (255, 255, 255), 2, 8, 0)
                cv2.line(images[0], (gfx_awal, gfy_awal), (gfx_akhir, gfy_akhir), (255, 255, 255), 1)
                cv2.imwrite(output_path, images[0])

    def classification(self):
        df = pd.read_csv('M1F2_PSAX.csv')
        X = df.drop('CLASS', axis=1)
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        temp = X.shape[0]
        filename = 'modelSVM'
        loaded_model = joblib.load(filename)
        model = pickle.dumps(loaded_model)
        prediction = pickle.loads(model)
        print(X[temp-1:temp])
        result = prediction.predict(X[temp-1:temp])

        with open("M1F2_PSAX.csv", "r") as data:
            lines = data.readlines()
            lines = lines[:-1]
        with open("M1F2_PSAX.csv", "w") as data:
            for line in lines:
                data.write(line)

        if result == 0:
            print("Tidak Normal")
            return 'Tidak Normal'
        else:
            print("Normal")
            return 'Normal'

    def frames2video(self, images):
        img_array = []
        for i, img in images.items():
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
        out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    def post(self):
        patient_id = request.form['patient_id']
        videofile = request.files['video']
        self.checked_at = datetime.datetime.now().date()

        filename = werkzeug.utils.secure_filename(videofile.filename)
        # filename = "./DatasetsPSAX/" + videofile.filename
        localstorage = './localstorage/'
        print("\nReceived image File name : " + videofile.filename)
        #variable konstan
        self.R = 65 #radius
        self.X1, self.Y1 = 0, 0 #centerpoint
        self.CCX, self.CCY = [0] * 100, [0] * 100

        self.jumlah = 12
        self.goodFeatures = [np.array([[]]) for _ in range(10)]
        self.GFcoordinates = {}
        self.valnorm = 0

        self.lengthDif = [[] for _ in range (9)]

        self.direction = np.zeros((self.jumlah * 2, 9), dtype=float)
        self.directionI = np.zeros((self.jumlah * 2, 9), dtype=float)

        self.jumlahFrame = 10
        self.frames = {}
        res ={}

        patientData = db_session.query(PatientData).filter(PatientData.id == patient_id).first()

        # bucket = storage_client.bucket(bucket_name)
        user_directory = f'{current_user.username}_data/'
        patient_directory = f'{patientData.patient_name}_data/'
        video_store_path = localstorage + user_directory + patient_directory + f'{self.checked_at}/'
        os.makedirs(video_store_path, exist_ok=True)
        # blob = bucket.blob(video_store_path + videofile.filename)
        # blob.upload_from_string(rawVideo)

        videofile.save(video_store_path + filename)
        video_link = video_store_path + filename

        self.frames = self.video2frames(video_link)
        print('frames'+str(len(self.frames)))
        rawImages = copy.deepcopy(self.frames)
        print('rawImages:' + str(len(rawImages)))
        #Preprocessing
        res = self.median_filter(rawImages[0])
        res = self.high_boost_filter(rawImages[0], res, 2.5)
        res = self.morph(res)
        res = self.thresholding(res)
        #Segmentation
        res = self.canny(res)
        res = self.region_filter(res)
        #ambil 1 frame untuk menympan nilai center point
        height, width = res.shape
        self.X1, self.Y1 = (width // 2), (height // 2)


        res = self.coLinear(res)

        res = self.triangleEquation(res)
        # cv2.imshow("Triangle Equation", res)
        # cv2.waitKey(0)

        GFcoordinates = self.GetGoodFeaturesPSAX(res)

        #Simpan nilai koordinat good feature
        for i in range(len(rawImages)) :
            self.goodFeatures[i] = self.goodFeatures[i].astype(np.float32)
            if i == 0:
                for j in range(self.jumlah * 2):
                    x = GFcoordinates[j][0]
                    y = GFcoordinates[j][1]
                    self.goodFeatures[i] = np.append(self.goodFeatures[i], np.array([x, y], dtype=np.float32))
                self.goodFeatures[i] = self.goodFeatures[i].reshape((self.jumlah * 2, 1, 2))

        self.opticalFlowCalcwithNormalization(rawImages, self.goodFeatures)

        #Visualisasi tracking
        visualFrames1 = copy.deepcopy(self.frames)
        res = self.track_visualization(visualFrames1, self.goodFeatures)
        visualFrames2 = copy.deepcopy(self.frames)
        self.track_visualization2(visualFrames2, self.goodFeatures)

        #Feature Extraction
        self.featureExtractionPSAX(self.goodFeatures)

        self.ExtractionMethod()
        res = self.frames2video(res)

        # blob = bucket.blob(user_directory + patient_directory + '/' + f'{self.checked_at}' + '/' + f'{patientData.patient_name}_result')
        # blob.upload_from_string(res)

        result = self.classification()
        inputData = HeartCheck(checkResult=result, video_path=video_link, checked_at=datetime.datetime.now(), patient=patientData)
        checkResult = []
        checkResult.append({
            'patientName' : patientData.patient_name,
            'checkResult' : result,
            'checkedAt' : datetime.datetime.now(),})
        db_session.add(inputData)
        db_session.commit()
        try:
            db_session.close()
            return make_response(jsonify({'data' : checkResult, 'message' : 'Patient Checked Succesfully'}), 201)
        except Exception as e:
            db_session.rollback()
            return make_response(jsonify(error="Patient failed to checked", details=str(e)), 409)

api.add_resource(RegisterUser, "/register", methods = ["POST"])
# api.add_resource(LoginUser, "/login", methods = ["POST"])
api.add_resource(UploadVideo, "/upload", methods=["POST"])
api.add_resource(Preprocessing, "/detectEchocardiography", methods=["POST"])
api.add_resource(InputPatientData, "/inputPatientData",  methods = ["POST"])
api.add_resource(GetPatientsData, "/getPatientsData",  methods = ["GET"])
api.add_resource(GetPatientCheckHistory, "/getPatientHistory",  methods = ["GET"])
api.add_resource(HelloWorld, "/")


with app.app_context():
    init_db()

if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(host="0.0.0.0", port=8080)