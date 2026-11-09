from django.shortcuts import render
from django.contrib.admin import site as admin_site
from django.contrib.admin.views.decorators import staff_member_required


@staff_member_required(login_url='/en/456-admin/login/')
def chat_ui(request):
    # Build the same context dict Django Admin pages use
    # so base_site.html renders correctly (title, has_permission, etc.)
    admin_ctx = admin_site.each_context(request)
    ctx = {
        **admin_ctx,
        'title': 'AI Assistant',
        'subtitle': None,
    }
    return render(request, 'ai_assistant/chat.html', ctx)
