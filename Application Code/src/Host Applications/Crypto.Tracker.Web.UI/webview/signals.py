# signals.py

from django.db.models.signals import post_save, post_migrate
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import Role, UserProfile

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
    else:
        # Check if a profile exists, if not create one
        UserProfile.objects.get_or_create(user=instance)

@receiver(post_migrate)
def create_system_user(sender, **kwargs):
    if sender.name == 'webview':  # Replace 'webview' with the name of your Django app
        user, created = User.objects.get_or_create(username='system', defaults={
            'first_name': 'System',
            'last_name': 'User',
            'email': 'system@example.com',  # Use a meaningful email address
        })

        if created:
            system_role = Role.get_system_role()
            UserProfile.objects.create(user=user, role=system_role)
        else:
            # If the system user already exists, ensure they have a profile
            UserProfile.objects.get_or_create(user=user, defaults={'role': Role.get_system_role()})
