from flask import Flask , render_template , request
import os
from model import OCR
app = Flask(__name__)

current_directory = os.getcwd()
upload_path = os.path.join(current_directory,"static/upload/")

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files["image"]
        file_name = uploaded_file.filename
        path_save = os.path.join(upload_path,file_name)
        uploaded_file.save(path_save)
        text = OCR(path_save,file_name)
        return render_template('index.html',upload=True,upload_image=file_name,text=text)

    return render_template('index.html',upload=False)

if __name__:
    app.run(debug=True)