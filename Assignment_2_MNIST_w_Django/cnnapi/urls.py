from django.conf import settings
from django.conf.urls.static import static
from cnnapi import views

from django.urls import path

app_name = 'cnnapi'

urlpatterns = [
    path('',views.index,name='homepage'),
    path('predictImage',views.predictImage,name='predictImage'),
    path('viewDataBase',views.viewDataBase,name='viewDataBase'),
    
     
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)