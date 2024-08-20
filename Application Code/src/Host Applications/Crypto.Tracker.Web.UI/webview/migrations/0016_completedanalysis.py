# Generated by Django 4.2.12 on 2024-08-03 15:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('webview', '0015_walletanalysis_walletnote_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='CompletedAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('note', models.TextField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(auto_now_add=True)),
                ('concluded_happened', models.BooleanField(default=False)),
                ('wallet_analysis', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='completed_analyses', to='webview.walletanalysis')),
            ],
            options={
                'db_table': 'dev_crypto_tracker"."completed_analysis',
                'indexes': [models.Index(fields=['wallet_analysis'], name='completed_a_wallet__060683_idx'), models.Index(fields=['name'], name='completed_a_name_6b53a9_idx'), models.Index(fields=['completed_at'], name='completed_a_complet_2efe2a_idx')],
                'unique_together': {('wallet_analysis', 'name')},
            },
        ),
    ]