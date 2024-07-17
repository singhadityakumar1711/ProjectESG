from django.urls import path
from .views import upload_pdf_app1
from .views import ai_summarized
from .views import upload_pdf_app2
from .views import ai_principle_checklist
urlpatterns = [
    path("upload_pdf_app1/", upload_pdf_app1, name="upload_pdf_app1"),
    path("esgSummarize/", ai_summarized, name = "ai_summarized"),
    path("upload_pdf_app2/", upload_pdf_app2, name="upload_pdf_app2"),
    path("esgAssess/", ai_principle_checklist, name = "ai_principle_checklist"),
]
