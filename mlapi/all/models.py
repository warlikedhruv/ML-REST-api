from django.db import models

# Create your models here.
from django.db import models


class File_all(models.Model):
    image = models.ImageField(upload_to='all_img/' ,max_length=500, blank=True, null=True)
