from django.urls import path
from .views import QueryView

urlpatterns = [path("summarize/", QueryView.as_view(), name="summarize")]
