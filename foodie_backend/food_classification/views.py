import datetime
from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.response import Response
from .models import FoodImage
from .serializers import FoodImageSerializer
from static import inference
from PIL import Image
import numpy as np
import io
import os, sys
from django.conf import settings

# Add the 'models' directory to the Python path
sys.path.append(os.path.join(settings.BASE_DIR, 'static'))

class FoodImageViewSet(viewsets.ModelViewSet):
    queryset = FoodImage.objects.all()
    serializer_class = FoodImageSerializer

    def create(self, request, *args, **kwargs):
        # Check if image is present in request
        if 'image' not in request.FILES:
            return Response({'error': 'No image file uploaded.'}, status=status.HTTP_400_BAD_REQUEST)
        
        image_data = request.FILES['image']
        
        # Check the file extension
        if not allowed_file(image_data.name):
            return Response({'error': 'Invalid file type. Only JPG, JPEG, and PNG are allowed.'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate a unique filename for the uploaded image
        image_filename = 'uploaded_image_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.' + image_data.name.split('.')[-1]
        
        # Absolute path to save the uploaded image
        image_directory = os.path.join(settings.MEDIA_ROOT, 'uploaded_images')
        os.makedirs(image_directory, exist_ok=True)  # Create the directory if it doesn't exist
        image_path = os.path.join(image_directory, image_filename)
        
        # Save the uploaded image
        with open(image_path, 'wb+') as destination:
            for chunk in image_data.chunks():
                destination.write(chunk)
        
        # Open and classify image
        image = Image.open(image_path)
        predicted_class = inference.classify(image.convert('RGB'))
        
        # Save image and predicted class
        food_image = FoodImage.objects.create(
            image=os.path.join('uploaded_images', image_filename),
            predicted_class=predicted_class
        )
        
        serializer = FoodImageSerializer(food_image, context={'request': request})
        
        return Response({
            'id': serializer.data['id'],
            'image':  serializer.data['image'],
            'predicted_class': predicted_class,
            'created_at': serializer.data['created_at'],
            'image_url':  serializer.data['image']
        }, status=status.HTTP_201_CREATED)

def allowed_file(filename):
    """Check if the file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
