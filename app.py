from flask import Flask , request, make_response, jsonify, send_file
from flask_restful import Resource, Api
from flask_cors import CORS
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from functools import wraps
import jwt, os, datetime, werkzeug
import numpy as np
import cv2


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
            videofile = request.files['image']
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
        output_dir = 'frames'
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video)
        target_frames = 10
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

    def median_filter(self, rawImages):
        median_filtered_images = {}
        output_dir = 'medianfiltered'
        os.makedirs(output_dir, exist_ok=True)
        for frame_count, image in rawImages.items():
            res = np.copy(image)
            kernelsize = 27
            res = cv2.medianBlur(image, kernelsize)
            median_filtered_images[frame_count] = res
            output_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
            cv2.imwrite(output_path, res)
        return median_filtered_images

    def high_boost_filter(self, source, lpf, kons):
        hbfImages = {}
        output_dir = 'highboost'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, (image, lpf) in enumerate(zip(source.items(), lpf.items())):
            #tupple image, lpf index ke 1 menyimpan bytes image sedangkan 0 menyimpan framecount, sehingga yang diakses disnini addalah index 1
            res = np.copy(image[1])
            #operasi manual highboost
            for i in range(image[1].shape[0]):
                for j in range(image[1].shape[1]):
                    lpf_rgb = lpf[1][i, j]
                    # src_rgb = image[1][i, j]

                    for k in range(3):  # 3 channels (B, G, R)
                        # val = kons * src_rgb[k] - lpf_rgb[k]
                        val = kons * lpf_rgb[k]
                        val = min(max(val, 0), 255)
                        res[i, j, k] = val
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            hbfImages[framecount] = res
            output_path = os.path.join(output_dir, f'frame_{framecount:04d}.png')
            cv2.imwrite(output_path, res)
        return hbfImages

    def morph(self, source):
        morphImages = {}
        output_dir = 'morphology'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in source.items():
            res = np.copy(image)
            ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12), (3,3))
            res = cv2.morphologyEx(image, cv2.MORPH_OPEN, ellipse)
            res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, ellipse)
            morphImages[framecount] = res
            output_path = os.path.join(output_dir, f'frame_{framecount:04d}.png')
            cv2.imwrite(output_path, res)
        return morphImages

    def thresholding(self, source):
        thresholded = {}
        output_dir = 'thresholding'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in source.items():
            res = np.copy(image)
            _, res = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY) #original at 90
            thresholded[framecount] = res
            output_path = os.path.join(output_dir, f'frame_{framecount:04d}.png')
            cv2.imwrite(output_path, res)
        return thresholded

    def canny(self, source):
        cannyFiltered = {}
        output_dir = 'canny'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in source.items():
            res = image.copy()
            res = cv2.Canny(image, 0, 255, 3)
            cannyFiltered[framecount] = res
            output_path = os.path.join(output_dir, f'frame_{framecount:04d}.png')
            cv2.imwrite(output_path, res)
        return cannyFiltered

    def region_filter(self, source):
        regionFiltered = {}
        output_dir = 'region'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in source.items():
            contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            res = np.zeros_like(image)
            output_path = os.path.join(output_dir, f'frame_{framecount:04d}.png')
            for i in range(len(contours)):
                if len(contours[i]) > self.R:
                    cv2.drawContours(res, contours, i, (255, 0, 0), 1)
                    regionFiltered[framecount] = res
                    cv2.imwrite(output_path, res)
        return regionFiltered


    def coLinear(self, source):
        colinearFiltered = {}
        output_dir = 'colinear'
        os.makedirs(output_dir, exist_ok=True)
        for framecount, image in source.items():
            contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            res = np.zeros_like(image)
            output_path = os.path.join(output_dir, f'frame_{framecount:04d}.png')
            data = [0] * len(contours)
            idk = 0
            for i, contour in enumerate(contours):
                if len(contour) > self.R * 2:
                    pt = contour[i][len(contour[i]) // 4]
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
                    colinearFiltered[framecount] = res
                    cv2.imwrite(output_path, res)
        return colinearFiltered

    def straightLine(self, x1, y1, x2, y2):
        x = x1 - x2
        if x == 0:
            m = 1e6
        else:
            m = (y1 - y2) / x
        b = y1 - m * x1
        return m, b

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





    def post(self):
        #variable konstan
        self.R = 71 #radius
        self.X1, self.Y1 = 0, 0 #centerpoint
        self.CCX, self.CCY = [0] * 100, [0] * 100
        rawImages = {}
        res ={}
        if(request.method == "POST"):
            videofile = request.files['video']
            rawVideo = werkzeug.utils.secure_filename(videofile.filename)
            print("\nReceived image File name : " + videofile.filename)
            print(videofile)
        rawImages = self.video2frames(rawVideo)
        res = self.median_filter(rawImages)
        res = self.high_boost_filter(rawImages, res, 1.5)
        res = self.morph(res)
        res = self.thresholding(res)
        res = self.canny(res)
        res = self.region_filter(res)
        #ambil 1 frame untuk menympan nilai center point
        height, width = res[0].shape
        self.X1, self.Y1 = (width // 2), (height // 2)
        res = self.coLinear(res)

        



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