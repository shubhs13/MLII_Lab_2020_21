from django import forms
from .models import MNIST

class ImgForm(forms.ModelForm):
    class Meta:
        model = MNIST
        fields = ['sampleImage']
        