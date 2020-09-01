from django.db import models

# Create your models here.

class MNIST(models.Model):
    sampleImage = models.FileField(upload_to='input/')
    mlResult = models.CharField(max_length=1000,null=True,blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    
