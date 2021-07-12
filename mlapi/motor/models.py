
from django.db import models

# Create your models here.
from django.db import models


class File_motor(models.Model):
    image = models.ImageField(upload_to='motor_img/' ,max_length=500, blank=True, null=True)
