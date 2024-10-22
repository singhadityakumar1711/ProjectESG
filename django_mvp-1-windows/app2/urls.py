from django.urls import path
from .views import file_upload
from .views import ask_query
from .views import handle_cleanups
from .views import global_placeholder

urlpatterns = [
    path("upload_file/", file_upload, name="file_upload"),
    path("query/", ask_query, name="ask_query"),
    path("clean_data/", handle_cleanups, name="handle_cleanups"),
    path("placeholder/", global_placeholder, name="global_placeholder"),
]
