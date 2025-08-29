import uuid
from django.contrib.auth.models import AbstractUser, BaseUserManager, Group, Permission
from django.db import models
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


class CustomUserManager(BaseUserManager):
    def create_user(self, username, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email field must be set")
        email = self.normalize_email(email)
        user = self.model(username=username, email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password=None, **extra_fields):
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_admin', True)
        extra_fields.setdefault('is_active', True)

        if not extra_fields.get('is_superuser'):
            raise ValueError("Superuser must have is_superuser=True.")
        if not extra_fields.get('is_admin'):
            raise ValueError("Superuser must have is_admin=True.")
        return self.create_user(username, email, password, **extra_fields)


class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=150, blank=True)
    last_name = models.CharField(max_length=150, blank=True)
    is_auth = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    is_super_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)

    # Avoid reverse accessor clashes
    groups = models.ManyToManyField(
        Group,
        related_name='customuser_set',  # Add custom related name
        blank=True,
        help_text='The groups this user belongs to.',
        related_query_name='customuser',
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name='customuser_set',  # Add custom related name
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='customuser',
    )

    # Removing is_staff – not needed if not using Django admin
    is_staff = None

    objects = CustomUserManager()

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email']

    def __str__(self):
        return self.username

class BaseMediaItem(models.Model):
    INPUT_TYPE_CHOICES = [
        ('upload', 'Upload'),
        ('camera', 'Camera'),
        ('url', 'URL'),
    ]

    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    location = models.CharField(max_length=255, blank=True)
    input_type = models.CharField(
        max_length=20,
        choices=INPUT_TYPE_CHOICES,
        default='upload',
    )
    uploaded_at = models.DateTimeField(default=timezone.now)

    class Meta:
        abstract = True


class UserImageModel(BaseMediaItem):
    image_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='images/')
    # Add file size and dimensions for better management
    file_size = models.PositiveIntegerField(null=True, blank=True, help_text="File size in bytes")
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)

    def save(self, *args, **kwargs):
        if self.image:
            self.file_size = self.image.size
            # Get image dimensions if needed
            try:
                from PIL import Image
                img = Image.open(self.image)
                self.width, self.height = img.size
            except:
                pass
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Image {self.image_id} by {self.user.username}"


class UserVideoModel(BaseMediaItem):
    video_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video = models.FileField(upload_to='videos/')
    # Add file size and duration for better management
    file_size = models.PositiveIntegerField(null=True, blank=True, help_text="File size in bytes")
    duration = models.DurationField(null=True, blank=True, help_text="Video duration")

    def save(self, *args, **kwargs):
        if self.video:
            self.file_size = self.video.size
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Video {self.video_id} by {self.user.username}"


class BaseMediaProcessingResult(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    image = models.ForeignKey(
        UserImageModel,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    video = models.ForeignKey(
        UserVideoModel,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(default=timezone.now)
    # Add processing status for better tracking
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    class Meta:
        abstract = True


class Prediction(BaseMediaProcessingResult):
    prediction_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prediction = models.TextField(blank=True)
    # Add raw prediction data for debugging
    raw_output = models.TextField(blank=True, help_text="Raw ML model output")
    confidence_score = models.FloatField(null=True, blank=True, help_text="Prediction confidence (0-1)")
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")
    
    # Feedback fields - MANDATORY for every prediction
    user_feedback = models.CharField(max_length=10, choices=[
        ('like', 'Like'),
        ('dislike', 'Dislike')
    ], null=True, blank=True)
    feedback_timestamp = models.DateTimeField(null=True, blank=True)
    feedback_comment = models.TextField(blank=True, help_text="Optional user comment about the prediction")
    feedback_required = models.BooleanField(default=True, help_text="Feedback is mandatory for this prediction")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction {self.prediction_id} by {self.user.username}"

    @property
    def is_accurate(self):
        """Returns True if user liked the prediction"""
        return self.user_feedback == 'like'

    @property
    def feedback_given(self):
        """Returns True if user has given feedback"""
        return self.user_feedback is not None

    @property
    def feedback_pending(self):
        """Returns True if feedback is required but not given"""
        return self.feedback_required and not self.feedback_given


class PredictionFeedback(models.Model):
    """Detailed feedback model for predictions"""
    feedback_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE, related_name='detailed_feedback')
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    
    # Feedback details
    is_correct = models.BooleanField(help_text="Was the prediction correct?")
    comment = models.TextField(blank=True, help_text="User suggestions for improvement")
    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Feedback for {self.prediction.prediction_id} by {self.user.username}"


class SystemMetrics(models.Model):
    """System performance metrics"""
    date = models.DateField(unique=True)
    total_predictions = models.IntegerField(default=0)
    successful_predictions = models.IntegerField(default=0)
    failed_predictions = models.IntegerField(default=0)
    average_processing_time = models.FloatField(default=0.0)
    average_confidence = models.FloatField(default=0.0)
    new_users = models.IntegerField(default=0)
    active_users = models.IntegerField(default=0)
    
    # Feedback metrics
    positive_feedback = models.IntegerField(default=0)
    negative_feedback = models.IntegerField(default=0)
    feedback_rate = models.FloatField(default=0.0)  # Percentage of predictions with feedback
    
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"Metrics for {self.date}"


class UserActivity(models.Model):
    """Track user activity for analytics"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    activity_type = models.CharField(max_length=50, choices=[
        ('login', 'Login'),
        ('logout', 'Logout'),
        ('prediction', 'Prediction'),
        ('feedback', 'Feedback'),
        ('upload', 'Upload'),
    ])
    timestamp = models.DateTimeField(default=timezone.now)
    details = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username} - {self.activity_type} at {self.timestamp}"
