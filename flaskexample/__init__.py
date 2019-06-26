from flask import Flask, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'flaskexample/uploads'
app.secret_key = 'My secret key'

from flaskexample import views

