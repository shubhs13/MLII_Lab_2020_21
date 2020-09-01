from django.contrib import admin
from django.urls import include,path

from django.conf import settings
from django.conf.urls.static import static
from basic import views



app_name = 'basic'

urlpatterns = [
     path('', views.home, name='home'),
     path('predict/',views.PredictPage.as_view(), name='predict'),
     
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)