from django.shortcuts import render
from .models import MNIST
from .forms import ImgForm
from django.views.generic import TemplateView

from django.core.files.storage import FileSystemStorage


import numpy as np

from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
# Create your views here.

sqModel = models.load_model('./mlModel/FFNN.h5')
cnnModel = models.load_model('./mlModel/CNN_MNIST.h5')

def home(request):
    obj = MNIST.objects.order_by('-created_at')
    context = {
        'obj': obj
    }
    return render(request, "basic/base.html",context)


class PredictPage(TemplateView):
    template_name = 'basic/predict.html'

    def get(self, request,*args, **kwargs):
        form = ImgForm()
        return render(request,self.template_name, {'form':form})

    def post(self, request):
        try:
            if request.method == 'POST':
                # model = MNIST()
                # obj = MNIST.objects.all()
                form = ImgForm(request.POST, request.FILES)
                if form.is_valid():
                    print("FORM IS VALID")
                    img = form.cleaned_data['sampleImage']
                    # print(img)
                    # model.sampleImage = img
                    fs = FileSystemStorage()
                    filePathName = fs.save(img.name,img)
                    filePathName =fs.url(filePathName)
                    testimage='.'+filePathName
                    img = load_img(testimage, target_size=(28, 28))
                    x = img_to_array(img)
                    x=x/255
                    x=x.reshape(1,28,28,1)
                    predict = sqModel.predict_classes(x)

                    predict = predict[0]

                    # model.mlResult = predict
                    # model.save()
                else:
                    print("FORM IS NOT VALID")
                    imgF = request.FILES['sampleImage']
                    # print(img2.name) 
                    print("IMAGE RECEIVED")
                    fs = FileSystemStorage()
                    filePathName = fs.save(imgF.name,imgF)
                    filePathName =fs.url(filePathName)
                    testimage='.'+filePathName
                    imgF = load_img(testimage, target_size=(28, 28))
                    x = img_to_array(imgF)
                    x=x/255
                    x=x.reshape(1,28,28,1)
                    predict = sqModel.predict_classes(x)

                    predict = predict[0]               
    
        except TypeError:
            print("Reached Exeption")
            form = ImgForm()
            img = 0
            predict = "Error"
            

        return render(request, self.template_name ,{'form':form,'predict': predict}) 
        
        # {
        #             'form':form,
        #             'filePathName':filePathName,
        #             'predict':predict,
        #             'obj':obj
        #             } )         