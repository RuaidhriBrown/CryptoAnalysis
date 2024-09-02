from django.contrib.auth.models import User
from django.db import models
from django.utils.timezone import make_aware, get_current_timezone
from datetime import datetime

# Define the default date
default_date_naive = datetime(2021, 5, 26, 11, 31, 0)
DEFAULT_LAST_UPDATED = make_aware(default_date_naive, timezone=get_current_timezone())

class Role(models.Model):
    SYSTEM_ROLE_NAME = 'System'  # Constant to refer to the system role

    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

    @staticmethod
    def get_system_role():
        """Retrieve the system role, creating it if it doesn't exist."""
        role, created = Role.objects.get_or_create(name=Role.SYSTEM_ROLE_NAME)
        return role

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.ForeignKey(Role, on_delete=models.SET_NULL, null=True, blank=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.user.username


class EthereumTransaction(models.Model):
    block_number = models.BigIntegerField()
    timestamp = models.DateTimeField()
    hash = models.CharField(max_length=66, primary_key=True)
    nonce = models.BigIntegerField()
    block_hash = models.CharField(max_length=66)
    transaction_index = models.IntegerField()
    from_address = models.CharField(max_length=42)
    to_address = models.CharField(max_length=42)
    value = models.BigIntegerField(null=True, blank=True)
    gas = models.BigIntegerField()
    gas_price = models.BigIntegerField()
    is_error = models.IntegerField()
    txreceipt_status = models.IntegerField(null=True, blank=True)
    input = models.TextField()
    contract_address = models.CharField(max_length=42, null=True, blank=True)
    cumulative_gas_used = models.BigIntegerField()
    gas_used = models.BigIntegerField()
    confirmations = models.BigIntegerField()
    method_id = models.CharField(max_length=10)
    function_name = models.CharField(max_length=1024)
    address = models.CharField(max_length=42)
    last_updated = models.DateTimeField(default=DEFAULT_LAST_UPDATED)

    class Meta:
        db_table = 'dev_crypto_tracker"."ethereum_transaction'
        indexes = [
            models.Index(fields=['from_address']),
            models.Index(fields=['to_address']),
            models.Index(fields=['address']),
            models.Index(fields=['hash']),
            models.Index(fields=['block_number']),
            models.Index(fields=['from_address', 'to_address'])
        ]

    def __str__(self):
        return self.hash

class ERC20Transaction(models.Model):
    composite_id = models.CharField(max_length=200, primary_key=True, default='')
    block_number = models.BigIntegerField()
    timestamp = models.DateTimeField()
    hash = models.CharField(max_length=66)
    nonce = models.BigIntegerField()
    block_hash = models.CharField(max_length=66)
    from_address = models.CharField(max_length=42)
    contract_address = models.CharField(max_length=42, null=True, blank=True)
    to_address = models.CharField(max_length=42)
    value = models.BigIntegerField(null=True, blank=True)
    token_name = models.CharField(max_length=1024)
    token_decimal = models.IntegerField()
    transaction_index = models.IntegerField()
    gas = models.BigIntegerField()
    gas_price = models.BigIntegerField()
    gas_used = models.BigIntegerField()
    cumulative_gas_used = models.BigIntegerField()
    input = models.TextField()
    confirmations = models.CharField(max_length=66)
    address = models.CharField(max_length=42)
    last_updated = models.DateTimeField(default=DEFAULT_LAST_UPDATED)

    class Meta:
        db_table = 'dev_crypto_tracker"."erc20_transaction'
        unique_together = ('hash', 'from_address', 'to_address')
        indexes = [
            models.Index(fields=['from_address']),
            models.Index(fields=['to_address']),
            models.Index(fields=['address']),
            models.Index(fields=['hash']),
            models.Index(fields=['block_number']),
            models.Index(fields=['from_address', 'to_address'])
        ]

    def __str__(self):
        return self.hash

    def save(self, *args, **kwargs):
        if not self.composite_id:
            self.composite_id = f'{self.hash}_{self.from_address}_{self.to_address}'
        super().save(*args, **kwargs)

class Wallet(models.Model):
    address = models.CharField(max_length=42, unique=True)
    balance = models.DecimalField(max_digits=30, decimal_places=18, null=True, blank=True)
    total_sent_transactions = models.IntegerField(default=0)
    total_received_transactions = models.IntegerField(default=0)
    unique_sent_addresses = models.IntegerField(default=0)
    unique_received_addresses = models.IntegerField(default=0)
    total_ether_sent = models.DecimalField(max_digits=30, decimal_places=18, default=0)
    total_ether_received = models.DecimalField(max_digits=30, decimal_places=18, default=0)
    total_erc20_sent = models.DecimalField(max_digits=30, decimal_places=18, default=0)
    total_erc20_received = models.DecimalField(max_digits=30, decimal_places=18, default=0)
    last_updated = models.DateTimeField(default=DEFAULT_LAST_UPDATED)

    class Meta:
        db_table = 'dev_crypto_tracker"."wallet'
        indexes = [
            models.Index(fields=['address']),
        ]

    def __str__(self):
        return self.address

class WalletAnalysis(models.Model):
    wallet = models.ForeignKey(Wallet, on_delete=models.CASCADE, related_name='analyses')
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='analyses')
    owner = models.CharField(max_length=255, blank=True, null=True)
    tag = models.CharField(max_length=255, blank=True, null=True)
    believed_illicit_count = models.IntegerField(default=0)
    believed_illicit = models.BooleanField(default=False)
    confirmed_illicit = models.BooleanField(default=False)
    believed_crime = models.CharField(max_length=255, blank=True, null=True)
    last_analyzed = models.DateTimeField(blank=True, null=True)
    updating_note = models.TextField(blank=True, null=True)

    class Meta:
        db_table = 'dev_crypto_tracker"."wallet_analysis'
        unique_together = ('wallet', 'user_profile')
        indexes = [
            models.Index(fields=['wallet']),
            models.Index(fields=['user_profile']),
        ]

    def __str__(self):
        return f"Analysis by {self.user_profile.user.username} on {self.wallet.address}"

    def save(self, *args, **kwargs):
        # Determine if this is the first time the object is being created
        is_new = self._state.adding
        
        # Call the original save method to save the WalletAnalysis instance
        super().save(*args, **kwargs)

        if is_new:
            # Create a system user if it doesn't exist
            system_user = self.get_or_create_system_user()

            # Create the note content based on the analysis creation
            note_content = (
                f"A new Wallet Analysis has been created for {self.wallet.address} by the system."
            )

            # Create and save the WalletNote associated with this new analysis
            WalletNote.objects.create(
                analysis=self,
                user=system_user,
                content=note_content,
                note_type='normal'  # You can define a special type for this, if needed
            )

    def get_or_create_system_user(self):
        """Retrieve the 'System' user, creating it if it doesn't exist."""
        system_role = Role.get_system_role()
    
        # Get or create the 'System' user
        system_user, created = User.objects.get_or_create(
            username='System',
            defaults={'is_active': False}
        )

        # Use get_or_create to ensure a UserProfile is created if it doesn't exist
        user_profile, profile_created = UserProfile.objects.get_or_create(
            user=system_user,
            defaults={'role': system_role}
        )

        if not profile_created and user_profile.role != system_role:
            # Update role if it does not match expected system role
            user_profile.role = system_role
            user_profile.save()

        return system_user

class WalletNote(models.Model):
    NOTE_TYPE_CHOICES = [
        ('completed_analysis', 'Completed Analysis'),
        ('normal', 'Normal'),
        # You can add more types if needed in the future
    ]

    analysis = models.ForeignKey(WalletAnalysis, on_delete=models.CASCADE, related_name='notes')
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='notes')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    note_type = models.CharField(max_length=50, choices=NOTE_TYPE_CHOICES, null=True, blank=True)  # New field for note type

    class Meta:
        db_table = 'dev_crypto_tracker"."wallet_note'
        indexes = [
            models.Index(fields=['analysis']),
            models.Index(fields=['user']),
            models.Index(fields=['created_at']),
            models.Index(fields=['note_type']),  # Index for note type to enable efficient filtering
        ]

    def __str__(self):
        return f"Note by {self.user} on {self.analysis.wallet.address} (Type: {self.get_note_type_display()})"


class CompletedAnalysis(models.Model):
    wallet_analysis = models.ForeignKey(WalletAnalysis, on_delete=models.CASCADE, related_name='completed_analyses')
    name = models.CharField(max_length=255)
    note = models.TextField(blank=True, null=True)
    completed_at = models.DateTimeField(auto_now_add=True)
    concluded_happened = models.BooleanField(default=False)

    class Meta:
        db_table = 'dev_crypto_tracker"."completed_analysis'
        unique_together = ('wallet_analysis', 'name')
        indexes = [
            models.Index(fields=['wallet_analysis']),
            models.Index(fields=['name']),
            models.Index(fields=['completed_at']),
        ]

    def __str__(self):
        return f"{self.name} analysis for {self.wallet_analysis.wallet.address}"

    def save(self, *args, **kwargs):
        # Call the original save method to save the CompletedAnalysis instance
        super().save(*args, **kwargs)

        # Ensure the 'System' user exists, or create it
        system_user = self.get_or_create_system_user()

        # Create the note content based on the analysis details
        note_content = (
            f"Completed Analysis: {self.name}\n"
            f"Concluded Happened: {'Yes' if self.concluded_happened else 'No'}\n"
            f"Details: {self.note or 'No additional details provided.'}"
        )

        # Create and save the WalletNote associated with this analysis
        WalletNote.objects.create(
            analysis=self.wallet_analysis,
            user=system_user,
            content=note_content,
            note_type='completed_analysis'  # Set the note type to 'completed_analysis'
        )

    def get_or_create_system_user(self):
        """Retrieve the 'System' user, creating it if it doesn't exist."""
        system_role = Role.get_system_role()
        
        # Get or create the 'System' user
        system_user, created = User.objects.get_or_create(
            username='System',
            defaults={'is_active': False}
        )

        # Ensure the 'System' user has a UserProfile, create it if it does not exist
        UserProfile.objects.get_or_create(
            user=system_user,
            defaults={'role': system_role}
        )
        
        return system_user
