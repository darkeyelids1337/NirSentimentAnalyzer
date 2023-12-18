from django.urls import path
from django.urls import re_path as url
from . import views

urlpatterns = [
    path('', views.home, name='sentiAnalyzer'),
    url(r'^predictsentiment', views.predict, name='predictsentiment'),
]