from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path('process_upload/', views.process_video, name='process_upload')
]
