from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.

from .serializers import ImageSerializer
from rest_framework import generics
import json
from .models import File_all
import tensorflow as tf
import cv2
import os
import numpy as np



class ImageViewSet(generics.ListAPIView):
    queryset = File_all.objects.all()
    serializer_class = ImageSerializer

    def post(self, request, *args, **kwargs):
        try:
            file = request.data['image']
            image = File_all.objects.get(id=1)
            image.image = file
            image.save()
            url = str(image.image.url)
            image_name = url.split('/')[-1]
            print(image_name)
            new_model = tf.keras.models.load_model('static/all.h5')

            #new_model.summary()
            IMG_SIZE = 100
            img_array = cv2.imread('media/all_img/'+ str(image_name))

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            X_train = np.array(new_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            # print(new_array)

            pred = new_model.predict(X_train)
            print(pred[0][0])
            return HttpResponse(json.dumps({'category': int(pred[0][0])}), status=200)
        except Exception as e:
            return HttpResponse(json.dumps({'error': str(e)}), status=404)