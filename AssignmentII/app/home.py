import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from app.mnist import model_predict
from app.cifar10 import model_predict_cnn, model_predict_ffd

bp = Blueprint('home',__name__)

@bp.route('/', methods=('GET','POST'))
def home():
    return render_template('home.html')


@bp.route('/mnist', methods=['GET','POST'])
def mnist():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f"app/static/img_upload/{f.filename}")
        print(f.filename)
        result = model_predict(f"app/static/img_upload/{f.filename}")
        return render_template('A2/mnist.html', img=f"img_upload/{f.filename}", value=result, output="Output")


    return render_template('A2/mnist.html', img="img/image_placeholder.jpg")

@bp.route('/cnn_cifar10', methods=['GET','POST'])
def cnn_cifar10():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f"app/static/img_upload/{f.filename}")
        print(f.filename)
        result = model_predict_cnn(f"app/static/img_upload/{f.filename}")
        return render_template('A2/cnn_cifar10.html', img=f"img_upload/{f.filename}", value=result, output="Output")


    return render_template('A2/cnn_cifar10.html', img="img/image_placeholder.jpg")

@bp.route('/ffd_cifar10', methods=['GET','POST'])
def ffd_cifar10():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f"app/static/img_upload/{f.filename}")
        print(f.filename)
        result = model_predict_ffd(f"app/static/img_upload/{f.filename}")
        return render_template('A2/ffd_cifar10.html', img=f"img_upload/{f.filename}", value=result, output="Output")


    return render_template('A2/ffd_cifar10.html', img="img/image_placeholder.jpg")