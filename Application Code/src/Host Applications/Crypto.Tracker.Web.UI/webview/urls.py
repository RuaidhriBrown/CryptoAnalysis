from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='home'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html', redirect_authenticated_user=True), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('profile/', views.profile, name='profile'),
    path('organization/', views.organization, name='organization'),
    path('documentation/', views.documentation, name='documentation'),
    path('statistics/', views.statistics, name='statistics'),
    
    path('transactions/', views.transaction_list, name='transaction_list'),
    path('transactions/<str:address>/', views.transactions_by_address, name='transactions_by_address'),
    path('wallets/', views.wallet_list, name='wallet_list'),
    path('update_wallets/', views.update_wallets, name='update_wallets'),
    path('run_analysis/', views.run_analysis, name='run_analysis'),
    path('wallets/<str:address>/', views.wallet_details, name='wallet_details'),
    path('wallets/<str:address>/update/', views.update_datasets, name='update_datasets'),
    path('wallets/<str:address>/download_linked_wallets/', views.download_linked_wallets, name='download_linked_wallets'),
    
    path('wallets/<int:wallet_id>/update_analysis/', views.update_wallet_analysis, name='update_wallet_analysis'),
    path('wallets/<int:wallet_id>/add_note/', views.add_wallet_note, name='add_wallet_note'),
    
    path('wallet/<int:wallet_id>/transaction-analysis/', views.transaction_analysis_results, name='transaction_analysis_results'),

    path('wallets/<int:wallet_id>/run_phishing_analysis_w_wallet/', views.run_phishing_detection_W_wallets, name='run_phishing_analysis_wallet'),
    path('wallets/<int:wallet_id>/run_phishing_analysis_W_transactions/', views.run_phishing_detection_W_transactions, name='run_phishing_analysis_transactions'),
    path('wallets/<int:wallet_id>/run_phishing_analysis_W_er20s/', views.run_phishing_detection_W_ER20, name='run_phishing_analysis_er20'),
    
    path('wallets/<int:wallet_id>/run_moneyLaundering_analysis_w_wallet/', views.run_moneyLaundering_detection_W_wallets, name='run_moneyLaundering_analysis_wallet'),
    path('wallets/<int:wallet_id>/run_moneyLaundering_analysis_W_transactions/', views.run_moneyLaundering_detection_W_transactions, name='run_moneyLaundering_analysis_transactions'),
    path('wallets/<int:wallet_id>/run_moneyLaundering_analysis_W_er20s/', views.run_moneyLaundering_detection_W_ER20, name='run_moneyLaundering_analysis_er20'),

    path('wallets/<int:wallet_id>/run_Money_Laundering_detection/', views.run_Money_Laundering_detection, name='run_Money_Laundering_detection'),

    
    path('run_phishing_analysis_all_wallets/', views.run_phishing_detection_all_wallets, name='run_phishing_analysis_all_wallets'),
    
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)