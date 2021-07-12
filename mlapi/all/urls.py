from django.urls import path
from .views import *

urlpatterns = [

    path('', ImageViewSet.as_view(), name='all'),
]