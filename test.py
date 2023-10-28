from flask import Flask, request, jsonify, send_file, send_from_directory
import bcrypt
import mysql.connector
import werkzeug
import os
app = Flask(__name__)


db = mysql.connector.connect(
    host="127.0.0.1",
    port="3306",
    user="root",
    password="root",
    database="db_cekjantung"
)

cursor = db.cursor()

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data['username']
    password = data['password'].encode('utf-8')
    hashed_password= bcrypt.hashpw(password, bcrypt.gensalt())
    
    sql = "INSERT INTO tb_users (user_name, user_password) VALUES (%s, %s)"
    values = (username, hashed_password.decode('utf-8'))

    try:
        cursor.execute(sql, values)
        db.commit()
        return jsonify(message="Registration successful"), 201
    except Exception as e:
        db.rollback()
        return jsonify(error="Registration failed", details=str(e)), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password'].encode('utf-8')  

    sql = "SELECT * FROM users WHERE username = %s"
    cursor.execute(sql, username)
    user = cursor.fetchone()

    if user:
        hashed_password = user[4] #Pada database password berada pada index ke 4
        if bcrypt.checkpw(password, hashed_password.encode('utf-8')):
            return jsonify(message="Login successful"), 200
        else:
            return jsonify(error="Login failed", details="Invalid credentials"), 401
    else:        
        return jsonify(error="Login failed", details="User not found"), 401


@app.route('/get_file/<filename>', methods=['GET'])
def get_file(filename):

    file_path = 'C:/Users/Jovan/Documents/testing/uploadedvideo/'  
    return send_file(os.path.join(file_path, filename))

@app.route('/get_file_list', methods=['GET'])
def get_file_list():
    file_directory = 'C:/Users/Jovan/Documents/testing/uploadedvideo/' 
    file_list = []
    
    for filename in os.listdir(file_directory):
        file_path = os.path.join(file_directory, filename)
        file_info = {
            'filename': filename,
            'size': os.path.getsize(file_path),
            'type': filename.split('.')[-1]  # Get the file type/extension
        }
        file_list.append(file_info)
    
    return jsonify(file_list)
	

@app.route('/upload', methods=["POST"])
def upload():
	if(request.method == "POST"):
		videofile = request.files['image']
		filename = werkzeug.utils.secure_filename(videofile.filename)
		print("\nReceived image File name : " + videofile.filename)
		videofile.save("./uploadedvideo/" + filename)
		return jsonify({
			"message" : "file uploaded successfully"
		})


if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run('0.0.0.0', port=9000)