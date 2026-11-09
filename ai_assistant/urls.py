from django.urls import path
from ai_assistant import api_views
from ai_assistant.views import chat_ui

urlpatterns = [
    path('chat-ui/', chat_ui, name='ai_chat_ui'),
    path('chat/', api_views.chat_view, name='ai_chat'),
    path('approve/', api_views.approve_view, name='ai_approve'),
]
