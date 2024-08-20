# Generated by Django 4.2.12 on 2024-08-03 14:54

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('webview', '0014_wallet_believed_count'),
    ]

    operations = [
        migrations.CreateModel(
            name='WalletAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('owner', models.CharField(blank=True, max_length=255, null=True)),
                ('believed_illicit_count', models.IntegerField(default=0)),
                ('believed_illicit', models.BooleanField(default=False)),
                ('confirmed_illicit', models.BooleanField(default=False)),
                ('believed_crime', models.CharField(blank=True, max_length=255, null=True)),
                ('last_analyzed', models.DateTimeField(blank=True, null=True)),
                ('updating_note', models.TextField(blank=True, null=True)),
            ],
            options={
                'db_table': 'dev_crypto_tracker"."wallet_analysis',
            },
        ),
        migrations.CreateModel(
            name='WalletNote',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('content', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'db_table': 'dev_crypto_tracker"."wallet_note',
            },
        ),
        migrations.RemoveIndex(
            model_name='wallet',
            name='wallet_owner_2cd6c8_idx',
        ),
        migrations.RemoveIndex(
            model_name='wallet',
            name='wallet_believe_bc7255_idx',
        ),
        migrations.RemoveIndex(
            model_name='wallet',
            name='wallet_confirm_6591ce_idx',
        ),
        migrations.RemoveField(
            model_name='wallet',
            name='believed_count',
        ),
        migrations.RemoveField(
            model_name='wallet',
            name='believed_crime',
        ),
        migrations.RemoveField(
            model_name='wallet',
            name='believed_illicit',
        ),
        migrations.RemoveField(
            model_name='wallet',
            name='confirmed_illicit',
        ),
        migrations.RemoveField(
            model_name='wallet',
            name='last_analyzed',
        ),
        migrations.RemoveField(
            model_name='wallet',
            name='notes',
        ),
        migrations.RemoveField(
            model_name='wallet',
            name='owner',
        ),
        migrations.AddField(
            model_name='walletnote',
            name='analysis',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='notes', to='webview.walletanalysis'),
        ),
        migrations.AddField(
            model_name='walletnote',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='notes', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='walletanalysis',
            name='user_profile',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='analyses', to='webview.userprofile'),
        ),
        migrations.AddField(
            model_name='walletanalysis',
            name='wallet',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='analyses', to='webview.wallet'),
        ),
        migrations.AddIndex(
            model_name='walletnote',
            index=models.Index(fields=['analysis'], name='wallet_note_analysi_9f7740_idx'),
        ),
        migrations.AddIndex(
            model_name='walletnote',
            index=models.Index(fields=['user'], name='wallet_note_user_id_53ca43_idx'),
        ),
        migrations.AddIndex(
            model_name='walletnote',
            index=models.Index(fields=['created_at'], name='wallet_note_created_e82223_idx'),
        ),
        migrations.AddIndex(
            model_name='walletanalysis',
            index=models.Index(fields=['wallet'], name='wallet_anal_wallet__e0f9a5_idx'),
        ),
        migrations.AddIndex(
            model_name='walletanalysis',
            index=models.Index(fields=['user_profile'], name='wallet_anal_user_pr_f6389d_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='walletanalysis',
            unique_together={('wallet', 'user_profile')},
        ),
    ]