from django.db import models

class ChatModel(models.Model):
    question = models.CharField(max_length= 250)
    answer = models.TextField()

    
