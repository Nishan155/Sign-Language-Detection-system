import re
from django import forms
from django.core.exceptions import ValidationError
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from django.contrib.auth import get_user_model
from .models import *

class CustomUserCreationForm(UserCreationForm):
    # Email field with specific widget attributes
    email = forms.EmailField(
        max_length=254,
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter your email'})
    )
    # First name field
    first_name = forms.CharField(
        max_length=150,
        required=False, # Often optional, adjust as needed
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter your first name'})
    )
    # Last name field
    last_name = forms.CharField(
        max_length=150,
        required=False, # Often optional, adjust as needed
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter your last name'})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add 'form-control' class and placeholders to username and password fields
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter username'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter password'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Confirm password'})
        # Add styling and placeholders for first_name and last_name
        self.fields['first_name'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter your first name'})
        self.fields['last_name'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter your last name'})

        # Remove default labels as placeholders provide necessary text
        self.fields['username'].label = ''
        self.fields['password1'].label = ''
        self.fields['password2'].label = ''
        self.fields['first_name'].label = ''
        self.fields['last_name'].label = ''


    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not email:
            raise ValidationError("This field is required.")

        # Convert to lowercase for consistent checking and storage
        email = email.lower()

        # Check if an email (case-insensitive) already exists in the database
        if CustomUser.objects.filter(email__iexact=email).exists():
            raise ValidationError("Email already taken")

        # Validate that the email is a Gmail address
        if not email.endswith('@gmail.com'):
            raise ValidationError("Email must be a Gmail address")

        return email

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if not username:
            raise ValidationError("This field is required.")

        # Check for minimum length
        if len(username) < 3:
            raise ValidationError("Username must be at least 3 characters long.")

        # Check for valid characters (letters, numbers, and underscores)
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            raise ValidationError("Username can only contain letters, numbers, and underscores.")

        return username

    def clean_password1(self):
        password = self.cleaned_data.get('password')
        if not password:
            raise ValidationError("This field is required.")

        # Check password length
        if len(password) < 8 or len(password) > 20:
            raise ValidationError("Password must be between 8 and 20 characters long.")

        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            raise ValidationError("Password must contain at least one uppercase letter.")

        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            raise ValidationError("Password must contain at least one lowercase letter.")

        # Check for at least one number
        if not re.search(r'\d', password):
            raise ValidationError("Password must contain at least one number.")

        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValidationError("Password must contain at least one special character.")

        return password

    def clean_password2(self):
        password1 = self.cleaned_data.get('password')
        password2 = self.cleaned_data.get('password2')

        # Check if both passwords are provided and if they match
        if password1 and password2 and password1 != password2:
            raise ValidationError("The two password fields must match.")

        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email'].lower()  # Save lowercase email
        user.first_name = self.cleaned_data.get('first_name')
        user.last_name = self.cleaned_data.get('last_name')
        user.set_password(self.cleaned_data['password'])  # Don't forget this line!

        if commit:
            user.save()
        return user


    class Meta(UserCreationForm.Meta):
        model = CustomUser
        # Include first_name and last_name in the fields for the form
        fields = ('username', 'email', 'first_name', 'last_name', 'password', 'password2')

class ContactForm(forms.Form):
    first_name = forms.CharField(
        max_length=30,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First Name'}),
        label='First Name'
    )
    last_name = forms.CharField(
        max_length=30,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last Name'}),
        label='Last Name'
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'}),
        label='Email'
    )
    message = forms.CharField(
        widget=forms.Textarea(attrs={'class': 'form-control', 'placeholder': 'Your Message', 'rows': 4}),
        label='Message'
    )

class AdminUserCreationForm(UserCreationForm):
    """Form for creating admin users"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter email address'
        })
    )
    
    first_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'First name'
        })
    )
    
    last_name = forms.CharField(
        max_length=30,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Last name'
        })
    )
    
    is_admin = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        help_text="Grant admin privileges to this user"
    )
    
    is_super_admin = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        help_text="Grant super admin privileges (only for admins)"
    )
    
    is_active = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        help_text="User can login to the system"
    )
    
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2', 
                 'is_admin', 'is_super_admin', 'is_active')
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add Bootstrap classes to form fields
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter username'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Enter password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm password'
        })
        
        # Add help text
        self.fields['username'].help_text = "Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only."
        self.fields['password1'].help_text = "Your password must contain at least 8 characters."
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exists():
            raise ValidationError("A user with this email already exists.")
        return email
    
    def clean(self):
        cleaned_data = super().clean()
        is_admin = cleaned_data.get('is_admin')
        is_super_admin = cleaned_data.get('is_super_admin')
        
        # Super admin can only be set if is_admin is True
        if is_super_admin and not is_admin:
            raise ValidationError("Super admin privilege requires admin privilege.")
        
        return cleaned_data
    
    def save(self, commit=True):
        user = super().save(commit=False)
        CustomUser.email = self.cleaned_data['email']
        CustomUser.first_name = self.cleaned_data['first_name']
        CustomUser.last_name = self.cleaned_data['last_name']
        CustomUser.is_admin = self.cleaned_data['is_admin']
        CustomUser.is_active = self.cleaned_data['is_active']
        
        if commit:
            CustomUser.save()
        return user


class UserUpdateForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'first_name', 'last_name', 'is_admin', 'is_active')
        widgets = {
            'username': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Username'
            }),
            'email': forms.EmailInput(attrs={
                'class': 'form-control',
                'placeholder': 'Email address'
            }),
            'first_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'First name'
            }),
            'last_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Last name'
            }),
            'is_admin': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
            'is_active': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            }),
        }
        
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if CustomUser.objects.filter(email=email).exclude(pk=self.instance.pk).exists():
            raise ValidationError("A user with this email already exists.")
        return email


class BulkActionForm(forms.Form):
    """Form for bulk actions on users/predictions"""
    action = forms.ChoiceField(
        choices=[
            ('', 'Select Action'),
            ('activate', 'Activate Users'),
            ('deactivate', 'Deactivate Users'),
            ('delete', 'Delete Users'),
            ('make_admin', 'Make Admin'),
            ('remove_admin', 'Remove Admin'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )
    
    selected_items = forms.CharField(
        widget=forms.HiddenInput()
    )
    
    confirm = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        }),
        help_text="I understand this action cannot be undone"
    )
    
    def clean(self):
        cleaned_data = super().clean()
        action = cleaned_data.get('action')
        confirm = cleaned_data.get('confirm')
        
        if action in ['delete', 'deactivate'] and not confirm:
            raise ValidationError("Please confirm this action.")
        
        return cleaned_data


class ReportReviewForm(forms.Form):
    """Form for reviewing low performance reports"""
    admin_notes = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Add your notes about this report...'
        }),
        required=False
    )
    
    mark_reviewed = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input'
        })
    )

