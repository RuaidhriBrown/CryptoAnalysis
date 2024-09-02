# signals.py

from django.db.models.signals import post_save, post_migrate
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import Role, UserProfile
import os


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    """
    Signal to create or update a user profile when a User is created or updated.
    """
    if created:
        UserProfile.objects.create(user=instance)
    else:
        # Ensure a profile exists, if not, create one.
        UserProfile.objects.get_or_create(user=instance)


@receiver(post_migrate)
def create_system_and_admin_users(sender, **kwargs):
    """
    Signal to create a system user and admin superuser after migrations.
    """
    if sender.name == 'webview':  # Replace 'webview' with the name of your Django app
        # Create or get the system user
        user, created = User.objects.get_or_create(
            username='system',
            defaults={
                'first_name': 'System',
                'last_name': 'User',
                'email': 'system@example.com',  # Use a meaningful email address
            }
        )

        if created:
            print(f'System user "{user.username}" created.')
        else:
            print(f'System user "{user.username}" already exists.')

        # Ensure the system role exists and assign it
        system_role, _ = Role.objects.get_or_create(name='System Admin')
        user_profile, profile_created = UserProfile.objects.get_or_create(
            user=user,
            defaults={'role': system_role}
        )

        if profile_created:
            print(f'Profile for system user "{user.username}" created with role "{system_role.name}".')
        else:
            print(f'Profile for system user "{user.username}" already exists.')

        # Create or get the admin superuser
        admin_username = os.getenv('DJANGO_SUPERUSER_USERNAME', 'admin')
        admin_email = os.getenv('DJANGO_SUPERUSER_EMAIL', 'admin@example.com')
        admin_password = os.getenv('DJANGO_SUPERUSER_PASSWORD', 'X1B2#WXYZ123a')

        admin_user, admin_created = User.objects.get_or_create(
            username=admin_username,
            defaults={
                'email': admin_email,
                'is_superuser': True,
                'is_staff': True,
            }
        )

        if admin_created:
            admin_user.set_password(admin_password)
            admin_user.save()
            print(f'Superuser "{admin_user.username}" created.')
        else:
            print(f'Superuser "{admin_user.username}" already exists.')
