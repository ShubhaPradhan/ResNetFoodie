from django.urls import path

from django.urls import path
from .views import FoodImageViewSet

urlpatterns = [
    path('classify-food/', FoodImageViewSet.as_view({'post': 'create'}), name='classify-food'),
]