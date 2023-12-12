from rest_framework import serializers
from . models import ChatModel



class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatModel
        fields = [
            'question',
            'answer,',
            ]