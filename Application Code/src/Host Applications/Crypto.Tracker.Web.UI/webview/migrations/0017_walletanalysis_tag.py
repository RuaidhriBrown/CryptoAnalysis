# Generated by Django 4.2.12 on 2024-08-17 12:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webview', '0016_completedanalysis'),
    ]

    operations = [
        migrations.AddField(
            model_name='walletanalysis',
            name='tag',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
