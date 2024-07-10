from flask import Flask , request, make_response, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS


from flask_security import Security, current_user, SQLAlchemySessionUserDatastore, roles_required, auth_token_required
from flask_security.utils import hash_password
from config import DevelopmentConfig
import boto3
import uuid

from functools import wraps
from database import db_session, init_db
from models import User, Role, PatientData, HeartCheck
import os, datetime, werkzeug, copy
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import processing

load_dotenv()
app = Flask(__name__)
api = Api(app)

app.config.from_object(DevelopmentConfig)

app.teardown_appcontext(lambda exc: db_session.close())

user_datastore = SQLAlchemySessionUserDatastore(db_session, User, Role)
security = Security(app, user_datastore)


s3 = boto3.client('s3',
                  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                  region_name=os.getenv("AWS_REGION"))

s3_bucket = os.getenv("AWS_BUCKET_NAME")

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
        # histories =  db_session.query(HeartCheck, PatientData).join(PatientData, HeartCheck.patient_id == PatientData.id).filter(HeartCheck.patient_id == patient_id)
        histories = db_session.query(HeartCheck).filter(HeartCheck.patient_id == patient_id)
        historyList = []
        if histories:
            for heart_check in histories:
                historyList.append({
                    'checkResult' : heart_check.checkResult,
                    'checkedAt' : heart_check.checked_at,
                    'videoPath' : heart_check.video_path
                })
        else:
            historyList.append("No History Data")
        return make_response(jsonify({
            'data' : historyList
        }), 201)




#Preprocessing, Segmentation, GoodFeature, Tracking and Feature Extraction
class Preprocessing(Resource):
    def __init__(self):
        self.R = 65  # radius
        self.X1, self.Y1 = 0, 0  # centerpoint
        self.CCX, self.CCY = [0] * 100, [0] * 100
        self.jumlah = 12
        self.goodFeatures = [np.array([[]]) for _ in range(10)]
        self.GFcoordinates = {}
        self.valnorm = 0
        self.lengthDif = [[] for _ in range(9)]
        self.direction = np.zeros((self.jumlah * 2, 9), dtype=float)
        self.directionI = np.zeros((self.jumlah * 2, 9), dtype=float)
        self.jumlahFrame = 10
        self.frames = {}


    def post(self):
        patient_id = request.form['patient_id']
        process_id = request.form['process_id']
        videofile = request.files['video']
        self.checked_at = datetime.datetime.now().date()

        filename = werkzeug.utils.secure_filename(videofile.filename)
        # filename = "./DatasetsPSAX/" + videofile.filename
        localstorage = './localstorage/'
        print("\nReceived image File name : " + videofile.filename)

        res ={}

        patientData = db_session.query(PatientData).filter(PatientData.id == patient_id).first()

        #save video temp in local for checking
        user_directory = f'{current_user.username}_data/'
        patient_directory = f'{patientData.patient_name}_data/'
        video_store_path = localstorage + user_directory + patient_directory + f'{self.checked_at}/'
        os.makedirs(video_store_path, exist_ok=True)

        videofile.save(video_store_path + filename)
        video_link = video_store_path + filename

        self.frames = processing.video2frames(jumlahFrame=self.jumlahFrame, video=video_link)
        print('frames'+str(len(self.frames)))
        rawImages = copy.deepcopy(self.frames)
        print('rawImages:' + str(len(rawImages)))
        #Preprocessing
        flow_choice = int(process_id)
        if flow_choice == 1:
            res = processing.gaussian_blur(rawImages[0], (5,5))
            res = processing.high_boost_filter(rawImages[0], res, 2.5)
            res = processing.morph(res)
            res = processing.adaptiveThresholding(res, 3, 1, 3, 2) #(blockSize=3, C=1, kernel=3, iterations=2)
        else :
            res = processing.median_filter(rawImages[0], 21)
            res = processing.high_boost_filter(rawImages[0], res, 2.5)
            res = processing.morph(res)
            res = processing.thresholding(res, 10, 255)


        #Segmentation
        res = processing.canny(res)
        res = processing.region_filter(R=self.R, image=res)
        #ambil 1 frame untuk menympan nilai center point
        height, width = res.shape
        self.X1, self.Y1 = (width // 2), (height // 2)


        res = processing.coLinear(R=self.R, CCX=self.CCX, CCY=self.CCY, X1=self.X1, Y1=self.Y1, image=res)

        res = processing.triangleEquation(R=self.R, CCX=self.CCX, CCY=self.CCY, X1=self.X1, Y1=self.Y1, source=res)
        # cv2.imshow("Triangle Equation", res)
        # cv2.waitKey(0)

        GFcoordinates, self.valnorm = processing.GetGoodFeaturesPSAX(jumlah=self.jumlah, valnorm=self.valnorm, res=res)

        #Simpan nilai koordinat good feature
        for i in range(len(rawImages)) :
            self.goodFeatures[i] = self.goodFeatures[i].astype(np.float32)
            if i == 0:
                for j in range(self.jumlah * 2):
                    x = GFcoordinates[j][0]
                    y = GFcoordinates[j][1]
                    self.goodFeatures[i] = np.append(self.goodFeatures[i], np.array([x, y], dtype=np.float32))
                self.goodFeatures[i] = self.goodFeatures[i].reshape((self.jumlah * 2, 1, 2))

        self.X1, self.Y1 = processing.findCenterPoint(R=self.R, source=res)

        processing.opticalFlowCalcwithNormalization(jumlah=self.jumlah, valnorm=self.valnorm, lengthDif =self.lengthDif, sources=rawImages, goodFeatures=self.goodFeatures)

        #Visualisasi tracking
        visualFrames1 = copy.deepcopy(self.frames)
        res = processing.track_visualization(jumlah=self.jumlah, images=visualFrames1, goodFeatures=self.goodFeatures)
        visualFrames2 = copy.deepcopy(self.frames)
        processing.track_visualization2(jumlah=self.jumlah, images=visualFrames2, goodFeatures=self.goodFeatures)

        #Feature Extraction
        processing.featureExtractionPSAX(jumlah=self.jumlah, X1=self.X1, Y1=self.Y1, direction=self.direction, goodFeatures=self.goodFeatures)

        processing.ExtractionMethod(jumlah=self.jumlah, direction=self.direction)

        result = processing.classification()

        unique_id = str(uuid.uuid1())[:8]
        res = processing.frames2video(res, unique_id)

        #save result on S3 bucket
        s3_path= user_directory + patient_directory + f'{self.checked_at}/' + f'result_{unique_id}.mp4'
        s3.upload_file(f'result_{unique_id}.mp4', s3_bucket, s3_path)
        video_link_s3 = f'https://{s3_bucket}.s3.amazonaws.com/{s3_path}'

        os.remove(f'result_{unique_id}.mp4')

        inputData = HeartCheck(checkResult=result, video_path=video_link_s3, checked_at=datetime.datetime.now(), patient=patientData)
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