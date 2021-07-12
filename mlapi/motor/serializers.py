from rest_framework import serializers
from .models import File_motor


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = File_motor
        fields = "__all__"