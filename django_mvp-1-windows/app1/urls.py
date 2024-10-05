from django.urls import path
from .views import ai_principle_checklist
from .views import upload_pdf
from .views import ai_chat_load
from .views import ai_chat_query
from .views import ai_summarized
from .views import ai_agent
from .views import pdf_history
urlpatterns = [
    path("esgSummarize/", ai_summarized, name = "ai_summarized"),
    path("upload_pdf/", upload_pdf, name="upload_pdf_app1"),
    path("esgAssess/", ai_principle_checklist, name = "ai_principle_checklist"),
    path("esgChatLoad/", ai_chat_load, name = "ai_chat_load"),
    path("esgChat/", ai_chat_query, name="ai_chat_query"),
    path("esgAgent/", ai_agent, name="ai_agent"),
    path("uploadHistory/", pdf_history, name="pdf_history")
]
