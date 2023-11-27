from flask import Flask , request, make_response, jsonify, send_file
from flask_restful import Resource, Api
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from functools import wraps
import jwt, os, datetime, werkzeug, copy
import numpy as np
import cv2
import math


from models import AuthModel

app = Flask(__name__)
api = Api(app)

#koneksi ke database
#app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/db_cekjantung'
#db = SQLAlchemy(app)
secret_key = os.urandom(24)
app.config['SECRET_KEY'] = secret_key

engine = create_engine("mysql+pymysql://root:root@localhost/db_cekjantung")

Session = sessionmaker(bind=engine)
session = Session()

class HelloWorld(Resource):
    def get(self):
        return "<p>Hello, World!</p>"

class RegisterUser(Resource):
    def post(self):
        usernameInput = request.json['user_name']
        useremailInput = request.json['user_email']
        passwordInput = request.json['user_password'].encode('utf-8')

        if usernameInput and passwordInput:
            user = session.query(AuthModel).filter_by(user_name=usernameInput).first()
            if not user:
                newUser = AuthModel(username=usernameInput, email=useremailInput)
                newUser.set_password(passwordInput)
                session.add(newUser)
                session.commit()
                try:
                    session.close()
                    return make_response(jsonify(message="Registration successful"), 201)
                except Exception as e:
                    session.rollback()
                    return make_response(jsonify(error="Registration failed", details=str(e)), )
            else:
                return make_response(jsonify(error="Username Telah Digunakan" ), 409  )

def token_requried(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token :
            return make_response(jsonify({'message': 'Token is missing'}), 401)
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return make_response(jsonify({'message': 'Token isexpired'}), 401)
        except jwt.InvalidTokenError:
            return make_response(jsonify({'message': 'Invalid Token'}), 401)
        return f(data, *args, **kwargs)
    return decorated

class LoginUser(Resource):
    def post(self):
        useremailInput = request.json['user_email']
        passwordInput = request.json['user_password'].encode('utf-8')

        user = session.query(AuthModel).filter_by(user_email=useremailInput).first()
        if user and user.check_password(passwordInput):
            #generate JWT token
            token = jwt.encode({
                'username' : user.user_name,
                'exp' : datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
            }, app.config['SECRET_KEY'], algorithm ='HS256')

            return make_response(jsonify({
                'username' : user.user_name,
                'message' :"Login Success",
                'token' : token
            }), 201)
        else :
            return make_response(jsonify(message="Login Failed"), 401)

class ProtectedLoginJWT(Resource):
    @token_requried
    def protected(self, data):
        return jsonify({'message': 'This is a protected endpoint', 'user': data['username']})

# class GetFileList(Resource):
#     # @token_requried
#     def get_file_list():
#         file_directory = './uploadedvideo/'
#         file_list = []

#         for filename in os.listdir(file_directory):
#             file_path = os.path.join(file_directory, filename)
#             file_info = {
#                 'filename': filename,
#                 'size': os.path.getsize(file_path),
#                 'type': filename.split('.')[-1]  # Get the file type/extension
#             }
#             file_list.append(file_info)

#         return jsonify(file_list)

# class GetFileName(Resource):
#     # @token_requried
#     def get_file(filename):
#         file_path = './uploadedvideo/'
#         return send_file(os.path.join(file_path, filename))

class UploadVideo(Resource):
    # @token_requried
    def upload_video():
        if(request.method == "POST"):
            videofile = request.files['video']
            filename = werkzeug.utils.secure_filename(videofile.filename)
            print("\nReceived image File name : " + videofile.filename)
            videofile.save("./uploadedvideo/" + filename)
            return jsonify({
                "message" : "file uploaded successfully"
            })


#Preprocessing and Segmentation
class Preprocessing(Resource):
    def video2frames(self, video):
        rawImages = {}
        output_dir = '1.frames'
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video)
        target_frames = self.jumlahFrame
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
                output_image_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
                cv2.imwrite(output_image_path, frame)
                frame_index += 1
            frame_count += 1
        cap.release()
        return rawImages

    def median_filter(self, image):
        output_dir = '2.medianfiltered'
        os.makedirs(output_dir, exist_ok=True)
        res = np.copy(image)
        kernelsize = 21
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
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (3,3))
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
                        if (abs(self.CCX[j] - pt1[0]) < 2) and (abs(self.CCY[j] - pt1[1]) < 2):
                            data[j] = 0
                        else:
                            data[j] = 1
        for i in range(len(contours)):
            if data[i] == 0:
                cv2.drawContours(res, contours, i, (255, 255, 255), 1, lineType=8, hierarchy=hierarchy, maxLevel=0, offset=(0, 0))
                output_path = os.path.join(output_dir, 'colinear.png')
                cv2.imwrite(output_path, res)
        return res

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

    def GetGoodFeaturesIntersection(self, res):
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
            self.valnorm = np.sqrt((rect_points[2][0] - rect_points[3][0]) ** 2 + (rect_points[2][1] - rect_points[3][1]) ** 2)

            # garis kanan
            garis = np.zeros(res.shape, dtype=res.dtype)
            cv2.line(garis, (int(rect_points[3][0]), int(rect_points[3][1])), (int(rect_points[0][0]), int(rect_points[0][1])), color)

            for y in range(garis.shape[0]):
                for x in range(garis.shape[1]):
                    if garis[y, x] > 0:
                        temp1 += 1
                        coordinate1[temp1][0] = (x, y)

            batasan = temp1
            data = float(temp1) / (self.jumlah + 1)
            temp1 = 0
            for i in np.arange(data / 2, batasan, data):
                temp1 += 1
                temp2 = int(round(i))
                coordinate2[temp1][0] = coordinate1[temp2][0]
                if temp1 == self.jumlah:
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
            data = float(temp1) / (self.jumlah + 1)
            temp1 = 0
            for i in np.arange(data / 2, batasan, data):
                temp1 += 1
                temp2 = int(round(i))
                coordinate2[temp1][0] = coordinate1[temp2][0]
                if temp1 == self.jumlah:
                    break

            garis = np.zeros(res.shape, dtype=res.dtype)
            temp1 = 0
            temp2 = self.jumlah

            for i in range(self.jumlah):
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
            self.valnorm = np.sqrt((rect_points[1][0] - rect_points[2][0]) ** 2 + (rect_points[1][1] - rect_points[2][1]) ** 2)
            print('kiri')

            garis = np.zeros(res.shape, dtype=res.dtype)
            cv2.line(garis, (int(rect_points[2][0]), int(rect_points[2][1])), (int(rect_points[3][0]), int(rect_points[3][1])), color)

            for x in range(garis.shape[1]):
                for y in range(garis.shape[0]):
                    if garis[y, x] > 0:
                        temp1 += 1
                        coordinate1[temp1][0] = (x, y)

            batasan = temp1
            data = float(temp1) / (self.jumlah + 1)
            temp1 = 0

            for i in np.arange(data / 2, batasan, data):
                temp1 += 1
                temp2 = int(round(i))
                coordinate2[temp1][0] = coordinate1[temp2][0]
                if temp1 == self.jumlah:
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
            data = float(temp1) / (self.jumlah + 1)
            temp1 = 0
            temp2 = 0

            for i in np.arange(data / 2, batasan, data):
                temp1 += 1
                temp2 = int(round(i))
                coordinate2[temp1][1] = coordinate1[temp2][1]
                if temp1 == self.jumlah:
                    break

            temp1 = 0
            temp2 = self.jumlah
            garis = np.zeros(res.shape, dtype=res.dtype)

            for i in range(1, self.jumlah + 1):
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
                    length[0] = np.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j][0][0]) ** 2 + (goodFeatures[i][j][0][1] - goodFeatures[i + 1][j][0][1]) ** 2) / self.valnorm * 100
                    if length[0] > thresh_diff:
                        if (j > 0 and j < 5) or (j > 6 and j < 11):
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
                        if j == 0 or j == 6:
                            length[3] = np.sqrt((goodFeatures[i][j + 1][0][0] - goodFeatures[i + 1][j + 1][0][0]) ** 2 + (goodFeatures[i][j + 1][0][1] - goodFeatures[i + 1][j + 1][0][1]) ** 2)
                            angleNorm = self.findAngle(goodFeatures[i][j + 1][0][0], goodFeatures[i][j + 1][0][1], goodFeatures[i + 1][j + 1][0][0], goodFeatures[i + 1][j + 1][0][1])
                            s = np.sin(angleNorm * np.pi / 180)
                            c = np.cos(angleNorm * np.pi / 180)
                            P = (goodFeatures[i][j][0][0] + s * length[3], goodFeatures[i][j][0][1] + c * length[3])
                            goodFeatures[i][j][0] = P

                        elif j == 5 or j == 11:
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

    def opticalFlowCalc(self, sources, goodFeatures):
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
            for j in range(self.jumlah * 2):
                output_path = os.path.join('Tracking', f'TrackingLK.png')
                gfx_awal = int(goodFeatures[0][j][0][0])
                gfy_awal = int(goodFeatures[0][j][0][1])
                gfx_akhir = int(goodFeatures[len(sources) - 1][j][0][0])
                gfy_akhir = int(goodFeatures[len(sources) - 1][j][0][1])
                cv2.line(sources[0], (gfx_awal, gfy_awal), (gfx_akhir, gfy_akhir), (255, 255, 255), 1)
                cv2.imwrite(output_path, sources[0])

                length = math.sqrt((gfx_awal - gfx_akhir)**2 + (gfy_awal - gfy_akhir)**2)
                self.lengthDif[i].append(length)

    def featureExtraction(self, goodFeatures):
        for j in range(self.jumlah):
            for i in range(9):
                # PENCARIAN SISI KIRI(GOODFEATURE) DERAJAT KEMIRINGAN
                a1 = math.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i + 1][j + self.jumlah][0][0]) ** 2 +
                            (goodFeatures[i][j][0][1] - goodFeatures[i + 1][j + self.jumlah][0][1]) ** 2)
                b1 = math.sqrt((goodFeatures[i + 1][j + self.jumlah][0][0] - goodFeatures[i][j + self.jumlah][0][0]) ** 2 +
                            (goodFeatures[i + 1][j + self.jumlah][0][1] - goodFeatures[i][j + self.jumlah][0][1]) ** 2)
                c1 = math.sqrt((goodFeatures[i][j + self.jumlah][0][0] - goodFeatures[i][j][0][0]) ** 2 +
                            (goodFeatures[i][j + self.jumlah][0][1] - goodFeatures[i][j][0][1]) ** 2)
                angle1 = math.acos((b1 ** 2 + c1 ** 2 - a1 ** 2) / (2 * b1 * c1)) * 180 / math.pi
                quadrant1 = (b1 ** 2 + c1 ** 2 - a1 ** 2) / (2 * b1 * c1) * 180 / math.pi
                if quadrant1 >= -1.27222e-14:
                    # MASUK
                    self.direction[j + self.jumlah][i] = int(1)
                    slope1 = (goodFeatures[i + 1][j + self.jumlah][0][1] - goodFeatures[i][j + self.jumlah][0][1]) / (
                            goodFeatures[i + 1][j + self.jumlah][0][0] - goodFeatures[i][j + self.jumlah][0][0])
                    slope2 = (goodFeatures[i][j][0][1] - goodFeatures[i][j + self.jumlah][0][1]) / (
                            goodFeatures[i][j][0][0] - goodFeatures[i][j + self.jumlah][0][0])
                    if slope1 > slope2:
                        print("MASUK ++", angle1)
                        self.directionI[j + self.jumlah][i] = int(1)
                    else:
                        print("MASUK --", angle1)
                        self.directionI[j + self.jumlah][i] = int(2)
                    # PEMBAGIAN EKSTRAKSI DILAKUKAN DISINI YA UNTUK BAGIAN YANG MASUK
                else:
                    # KELUAR
                    self.direction[j + self.jumlah][i] = int(0)
                    slope1 = (goodFeatures[i + 1][j + self.jumlah][0][1] - goodFeatures[i][j + self.jumlah][0][1]) / (
                            goodFeatures[i + 1][j + self.jumlah][0][0] - goodFeatures[i][j + self.jumlah][0][0])
                    slope2 = (goodFeatures[i][j][0][1] - goodFeatures[i][j + self.jumlah][0][1]) / (
                            goodFeatures[i][j][0][0] - goodFeatures[i][j + self.jumlah][0][0])
                    if slope1 < slope2:
                        print("KELUAR --", angle1)
                        self.directionI[j + self.jumlah][i] = int(3)
                    else:
                        print("KELUAR ++", angle1)
                        self.directionI[j + self.jumlah][i] = int(4)
                    # PEMBAGIAN EKSTRAKSI DILAKUKAN DISINI YA UNTUK BAGIAN YANG MASUK

                # PENCARIAN SISI KANAN (GOODFEATURE) DERAJAT KEMIRINGAN
                a2 = math.sqrt((goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0]) ** 2 +
                            (goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1]) ** 2)
                b2 = math.sqrt((goodFeatures[i][j + self.jumlah][0][0] - goodFeatures[i + 1][j][0][0]) ** 2 +
                            (goodFeatures[i][j + self.jumlah][0][1] - goodFeatures[i + 1][j][0][1]) ** 2)
                c2 = math.sqrt((goodFeatures[i][j][0][0] - goodFeatures[i][j + self.jumlah][0][0]) ** 2 +
                            (goodFeatures[i][j][0][1] - goodFeatures[i][j + self.jumlah][0][1]) ** 2)
                angle2 = math.acos((c2 ** 2 + a2 ** 2 - b2 ** 2) / (2 * a2 * c2)) * 180 / math.pi
                quadrant2 = (c2 ** 2 + a2 ** 2 - b2 ** 2) / (2 * a2 * c2) * 180 / math.pi
                if quadrant2 >= -1.27222e-14:
                    # MASUK
                    self.direction[j][i] = int(1)
                    slope1 = (goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1]) / (
                            goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0])
                    slope2 = (goodFeatures[i][j + self.jumlah][0][1] - goodFeatures[i][j][0][1]) / (
                            goodFeatures[i][j + self.jumlah][0][0] - goodFeatures[i][j][0][0])
                    if slope1 < slope2:
                        print("MASUK --", angle2)
                        self.directionI[j][i] = int(1)
                    else:
                        print("MASUK ++", angle2)
                        self.directionI[j][i] = int(2)
                else:
                    # KELUAR
                    self.direction[j][i] = int(0)
                    slope1 = (goodFeatures[i + 1][j][0][1] - goodFeatures[i][j][0][1]) / (
                            goodFeatures[i + 1][j][0][0] - goodFeatures[i][j][0][0])
                    slope2 = (goodFeatures[i][j + self.jumlah][0][1] - goodFeatures[i][j][0][1]) / (
                            goodFeatures[i][j + self.jumlah][0][0] - goodFeatures[i][j][0][0])
                    if slope1 > slope2:
                        print("KELUAR ++", angle2)
                        self.directionI[j][i] = int(3)
                    else:
                        print("KELUAR --", angle2)
                        self.directionI[j][i] = int(4)


    def ExtractionMethod(self):
        pf, nf, pm, nm = [[] for _ in range(self.jumlah * 2)], [[] for _ in range(self.jumlah * 2)], [[] for _ in range(self.jumlah * 2)], [[] for _ in range(self.jumlah * 2)]
        for j in range(self.jumlah * 2):
            num1, num2, num3, num4 = 0, 0, 0, 0
            for i in range(9):
                if self.direction[j][i] == 1:
                    num1 += 1
                    num3 += self.lengthDif[i][j]
                else:
                    num2 += 1
                    num4 += self.lengthDif[i][j]
            pf[j] = num1 / 9
            nf[j] = num2 / 9
            pm[j] = num3
            nm[j] = num4

        # MENYIMPAN FEATURE EXTRACTION METHOD I
        with open("M1F1_2AC.csv", "a") as myfile:
            for j in range((self.jumlah * 2) - 1):
                myfile.write(f"{str(pf[j])},{str(nf[j])},{str(pm[j])},{str(nm[j])},")
                if j == (self.jumlah * 2) - 2:
                    myfile.write(f"{str(pf[j + 1])},{str(nf[j + 1])},{str(pm[j + 1])},{str(nm[j + 1])}\n")

    def sort_by_second(self, elem):
        return elem[1]

    def track_visualization(self, images, goodFeatures):
        output_dir = '10.Tracking'
        os.makedirs(output_dir, exist_ok=True)

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
            output_path = os.path.join(output_dir, f"tracking_{i}.png")
            cv2.imwrite(output_path, image)

    def triangleEquation(self, source):

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
            return res

        if jum2 == 1:
            print("bentuk=2")
            j = 0
            for m in range(len(contours)):
                if len(contours[m]) > self.R:
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
                        if d <= 15 :
                            data1[k] = i + 3
                            k += 1
                            self.CCX[j] = p4[0]
                            self.CCY[j] = p4[1]
                            cv2.line(source, (int(self.CCX[j] - 1), int(self.CCY[j])), (int(self.CCX[j] + 1), int(self.CCY[j])), (255, 0, 0), thickness=1)
                            cv2.line(source, (int(self.CCX[j]), int(self.CCY[j] - 1)), (int(self.CCX[j]), int(self.CCY[j] + 1)), (255, 0, 0), thickness=1)

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
                    jum = 0
                    min = 2000.0
                    p1 = contours[m][data1[0]][0]
                    for i in range(data1[1], data1[1] + self.R):
                        p = contours[m][i][0]
                        a1 = np.sqrt(pow((p1[0] - p[0]), 2.0) + pow((p1[1] - p[1]), 2.0))
                        b1 = np.sqrt(pow((center[0] - p[0]), 2.0) + pow((center[1] - p[1]), 2.0))
                        c1 = np.sqrt(pow((center[0] - p1[0]), 2.0) + pow((center[1] - p1[1]), 2.0))
                        alpha = math.acos((b1 * b1 + c1 * c1 - a1 * a1)/ (2 * b1 * c1)) * 180/math.pi
                        print(alpha)
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
                        alpha = math.acos((b1 * b1 + c1 * c1 - a1 * a1)/ (2 * b1 * c1)) * 180/math.pi
                        print(alpha)
                        if (alpha < min):
                            min = alpha
                            jum1 = i

                    p1 = contours[m][jum][0]
                    p2 = contours[m][jum1][0]
                    x1[0][0] = p1[0]
                    y1[0][0] = p1[1]
                    x2[0][0] = p2[0]
                    y2[0][0] = p2[1]
                    cv2.line(source, (int(x1[0][0]), int(y1[0][0])), (int(x2[0][0]), int(y2[0][0])), (255, 0, 0), thickness=1)

                j += 1

            contours, hierarchy = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            res = np.zeros_like(source)
            for m in range(len(contours)):
                if len(contours[m]) > self.R:
                    cv2.drawContours(res, contours, m, (255, 0, 0), 1, lineType=8,)

            cv2.line(source, (int(x1[0][0]), int(y1[0][0])), (int(x2[0][0]), int(y2[0][0])), (255, 255, 255), thickness=1)


            # contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for m in range(len(contours)):
            #     if len(contours[m]) > self.R:
            #         pt = (self.X1, self.Y1)
            #         out = cv2.pointPolygonTest(contours[m], pt, False)
            #         if out > 0:
            #             break
            # for m in range(len(contours)):
            #     cv2.drawContours(res, contours, m (255, 0, 0), 1, lineType=8,)
            return res
        if jum2 == 2:
            print("bentuk=3")
        if jum2 == 3:
            print("bentuk=4")
        if jum2 == 4 or jum2 == 5:
            print("bentuk=5")
    def post(self):
        #variable konstan
        self.R = 65 #radius
        self.X1, self.Y1 = 0, 0 #centerpoint
        self.CCX, self.CCY = [0] * 100, [0] * 100

        self.jumlah = 6
        self.goodFeatures = [np.array([[]]) for _ in range(10)]
        self.GFcoordinates = {}
        self.valnorm = 0

        self.lengthDif = [[] for _ in range (9)]

        self.direction = np.zeros((12, 9), dtype=float)
        self.directionI = np.zeros((12, 9), dtype=float)

        self.jumlahFrame = 10
        self.frames = {}
        res ={}
        if(request.method == "POST"):
            videofile = request.files['video']
            rawVideo = werkzeug.utils.secure_filename(videofile.filename)
            print("\nReceived image File name : " + videofile.filename)
            print(videofile)
        self.frames = self.video2frames(rawVideo)
        rawImages = copy.copy(self.frames)
        #Preprocessing
        res = self.median_filter(rawImages[0])
        res = self.high_boost_filter(rawImages[0], res, 1.5)
        res = self.morph(res)
        res = self.thresholding(res)
        #Segmentation
        res = self.canny(res)
        res = self.region_filter(res)
        #ambil 1 frame untuk menympan nilai center point
        height, width = res.shape
        self.X1, self.Y1 = (width // 2), (height // 2)

         # Visualisasi titik tengah

        res = self.coLinear(res)

        # cv2.circle(res, (self.X1 ,self.Y1), 1, (255, 255, 255), 2, 8, 0)
        # cv2.imshow(f'nyoba', res)
        # cv2.waitKey(0)

        res = self.triangleEquation(res)
        # cv2.imshow("Triangle Equation", res)
        # cv2.waitKey(0)

        #Tracking
        GFcoordinates = self.GetGoodFeaturesPSAX(res)
        #Simpan nilai koordinat good feature
        for i in range(len(rawImages)) :
            self.goodFeatures[i] = self.goodFeatures[i].astype(np.float32)
            # for j in range(1, self.jumlah * 2 + 1):
            if i == 0:
                for j in range(self.jumlah * 2):

                    # x = GFcoordinates[j][2][0]
                    # y = GFcoordinates[j][2][1]

                    #PSAX
                    x = GFcoordinates[j][0]
                    y = GFcoordinates[j][1]
                    self.goodFeatures[i] = np.append(self.goodFeatures[i], np.array([x, y], dtype=np.float32))
                self.goodFeatures[i] = self.goodFeatures[i].reshape((self.jumlah * 2, 1, 2))

        #Visualisasi Good Feature
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

        self.opticalFlowCalc(rawImages, self.goodFeatures)

        #Feature Extraction
        self.featureExtraction(self.goodFeatures)
        self.track_visualization(self.frames, self.goodFeatures)

        output_dir = '11.Output'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in self.frames.items():
            output_path = os.path.join(output_dir, f'frame_{framecount}.png')
            cv2.imwrite(output_path, image)

        self.ExtractionMethod()

api.add_resource(RegisterUser, "/register", methods = ["POST"])
api.add_resource(LoginUser, "/login", methods = ["POST"])
api.add_resource(ProtectedLoginJWT, "/protected", methods=["GET"])
#api.add_resource(GetFileName, "/get_file/<filename>", methods=["GET"])
#api.add_resource(GetFileList, "/get_file_list", methods=["GET"])
api.add_resource(UploadVideo, "/upload", methods=["POST"])
api.add_resource(Preprocessing, "/preprocessing", methods=["POST"])
api.add_resource(HelloWorld, "/")

if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    # app.run(port='9000', debug=True)
    app.run('0.0.0.0')