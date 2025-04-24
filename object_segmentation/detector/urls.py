from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detection/', views.detection, name='detection'),
    path('segmentation/', views.segmentation, name='segmentation'),
]