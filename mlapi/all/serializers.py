from rest_framework import serializers
from .models import File_all


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = File_all
        fields = "__all__"