from django.urls import path
from .views import ProcessApiView



urlpatterns = [
    path('process/', ProcessApiView.as_view(), name='procss'),
]