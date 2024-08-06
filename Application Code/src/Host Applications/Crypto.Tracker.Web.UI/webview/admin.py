from django.contrib import admin
from .models import (
    EthereumTransaction,
    ERC20Transaction,
    Wallet,
    WalletAnalysis,
    WalletNote,
    Role,
    UserProfile,
)

@admin.register(EthereumTransaction)
class EthereumTransactionAdmin(admin.ModelAdmin):
    list_display = ('address', 'hash', 'block_number', 'timestamp', 'from_address', 'to_address', 'value')
    search_fields = ('hash', 'from_address', 'to_address', 'address')
    list_filter = ('block_number', 'timestamp', 'address')

@admin.register(ERC20Transaction)
class ERC20TransactionAdmin(admin.ModelAdmin):
    list_display = ('address', 'hash', 'block_number', 'timestamp', 'from_address', 'to_address', 'value', 'token_name')
    search_fields = ('hash', 'from_address', 'to_address', 'address', 'token_name')
    list_filter = ('block_number', 'timestamp', 'address', 'token_name')

@admin.register(Wallet)
class WalletAdmin(admin.ModelAdmin):
    list_display = ('address', 'total_sent_transactions', 'total_received_transactions', 'last_updated')
    search_fields = ('address',)
    list_filter = ('last_updated',)

@admin.register(WalletAnalysis)
class WalletAnalysisAdmin(admin.ModelAdmin):
    list_display = ('wallet', 'user_profile', 'owner', 'believed_illicit', 'confirmed_illicit', 'last_analyzed')
    search_fields = ('wallet__address', 'user_profile__user__username', 'owner')
    list_filter = ('believed_illicit', 'confirmed_illicit', 'last_analyzed')

@admin.register(WalletNote)
class WalletNoteAdmin(admin.ModelAdmin):
    list_display = ('analysis', 'user', 'created_at', 'updated_at')
    search_fields = ('analysis__wallet__address', 'user__username', 'content')
    list_filter = ('created_at', 'updated_at')

@admin.register(Role)
class RoleAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'phone_number')
    search_fields = ('user__username', 'role__name', 'phone_number')
    list_filter = ('role',)
