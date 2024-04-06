from rest_framework import serializers
from .models import FoodImage

class FoodImageSerializer(serializers.ModelSerializer):

    image_url = serializers.SerializerMethodField()

    class Meta:
        model = FoodImage
        fields = ('id', 'image', 'predicted_class', 'created_at', 'image_url')

    def get_image_url(self, obj):
        return self.context['request'].build_absolute_uri(obj.image.url)