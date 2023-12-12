from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import QueryChain
urlpatterns = [
    #path('login/',MyloginAPI.as_view()),
    #path('register/',Register.as_view()),
    path('querychain/',QueryChain.as_view()),
    #path('file/', file.as_view()),
]
# + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
# urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)*