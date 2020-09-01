from flask import Flask,render_template,request
import sys
sys.path.insert(1,'assignments')
from assignment1 import models
import numpy as np

model = models()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('first_page.html')

@app.route('/lab1_fp')
def lab1_fp():
    return render_template('assignment1/fp.html')

@app.route('/mnist_test',methods=['POST','GET'])
def test_mnist():
    if request.method == 'POST':
        classes = [str(i) for i in range(10)]
        f = request.files['file']
        fname = f.filename
        f.save(f'static/saved_images/{fname}')
        value = model.predict_mnist(f'static/saved_images/{fname}')
        value = value * 100
        value = np.array(value,dtype=np.int8)
        print(value)

        return render_template('assignment1/mnist.html',value=value,classes=classes,img=f'saved_images/{fname}',op='Output',result=np.argmax(value))
    return render_template('assignment1/mnist.html',img='img/image_placeholder.jpg')

@app.route('/cifar10_test',methods=['POST','GET'])
def test_cifar10():
    if request.method == 'POST':
        classes = ['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        f = request.files['file']
        fname = f.filename
        f.save(f'static/saved_images/{fname}')
        value = model.predict_cifar10(f'static/saved_images/{fname}')
        print(value)
        value = (value * 200) // 2
        value = np.array(value,dtype=np.int8)
        

        return render_template('assignment1/cifar10.html',value=value,classes=classes,img=f'saved_images/{fname}',op='Output',result=classes[np.argmax(value)])
    return render_template('assignment1/cifar10.html',img='img/image_placeholder.jpg')

@app.route('/cifar10_cnn_test',methods=['POST','GET'])
def test_cifar10_cnn():
    if request.method == 'POST':
        if request.files['file'].filename != '':
            classes = ['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            f = request.files['file']
            fname = f.filename
            f.save(f'static/saved_images/{fname}')
            value = model.predict_cifar10_cnn(f'static/saved_images/{fname}')
            print(value)
            value = (value * 200) // 2
            value = np.array(value,dtype=np.int8)

            return render_template('assignment1/cifar10_cnn.html',value=value,classes=classes,img=f'saved_images/{fname}',op='Output',result=classes[np.argmax(value)])
    return render_template('assignment1/cifar10_cnn.html',img='img/image_placeholder.jpg')

@app.route('/mnist_report')
def mnist_report():
    (precision,recall,f1) = model.create_model_report('mnist')
    print(precision)
    print(recall)
    print(f1)
    return render_template('assignment1/mnist_report.html',p=str(precision),r=str(recall),f=str(f1))

@app.route('/cifar10_report')
def cifar10_report():
    (precision,recall,f1) = model.create_model_report('cifar10')
    print(precision)
    print(recall)
    print(f1)
    return render_template('assignment1/cifar10_report.html',p=str(precision),r=str(recall),f=str(f1))

@app.route('/cifar10_cnn_report')
def cifar10_cnn_report():
    (precision,recall,f1) = model.create_model_report('cifar10_cnn')
    print(precision)
    print(recall)
    print(f1)
    return render_template('assignment1/cifar10_cnn_report.html',p=str(precision),r=str(recall),f=str(f1))

if __name__ == "__main__":
    app.run(debug=True)
