from django.urls import path
from .views import upload_pdf
from .views import ai_summarized
urlpatterns = [
    path("upload_pdf/", upload_pdf, name="upload_pdf"),
    path("esgSummarize/", ai_summarized, name = "ai_summarized"),
]
