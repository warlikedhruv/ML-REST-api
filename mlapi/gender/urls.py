from django.urls import path
from .views import *

urlpatterns = [
    path('1', FileUploadView.as_view()),
    path('', ImageViewSet.as_view(), name='upload'),
]