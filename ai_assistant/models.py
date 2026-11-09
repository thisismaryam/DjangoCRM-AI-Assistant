
from django.db import models
from django.conf import settings


class ActionRequest(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             on_delete=models.CASCADE)
    action_type = models.CharField(max_length=50)  # e.g., 'CREATE_LEAD'
    payload = models.JSONField()  # The data the AI wants to save
    status = models.CharField(
        max_length=10, choices=STATUS_CHOICES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.action_type} - {self.status}"
