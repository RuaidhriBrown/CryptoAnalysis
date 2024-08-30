# Generated by Django 4.2.12 on 2024-08-18 17:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webview', '0017_walletanalysis_tag'),
    ]

    operations = [
        migrations.AddField(
            model_name='walletnote',
            name='note_type',
            field=models.CharField(blank=True, choices=[('completed_analysis', 'Completed Analysis'), ('normal', 'Normal')], max_length=50, null=True),
        ),
        migrations.AddIndex(
            model_name='walletnote',
            index=models.Index(fields=['note_type'], name='wallet_note_note_ty_3338b9_idx'),
        ),
    ]
