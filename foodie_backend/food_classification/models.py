# Create your models here.
from django.db import models
import os
from uuid import uuid4

def image_file_path(instance, filename):
    """Generate file path for new image"""
    ext = filename.split('.')[-1]
    filename = f'{uuid4()}.{ext}'
    return os.path.join('uploads/', filename)

class FoodImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    predicted_class = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.predicted_class