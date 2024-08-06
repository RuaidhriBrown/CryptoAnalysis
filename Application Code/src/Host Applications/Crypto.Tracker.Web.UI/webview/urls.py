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
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
