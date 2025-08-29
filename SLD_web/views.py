# Standard library imports
import os
import uuid
import base64
import logging
import time, json, datetime
from datetime import timedelta

# Third-party imports
import cv2
import subprocess
import numpy as np
import mediapipe as mp

# Django core imports
from django.conf import settings
from django.contrib import messages
from django.utils.crypto import get_random_string
from django.contrib.auth import login, logout, get_user_model, update_session_auth_hash
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.forms import PasswordChangeForm, PasswordResetForm
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.views import LoginView
from django.contrib.auth.forms import SetPasswordForm
from django.contrib.messages.views import SuccessMessageMixin
from django.contrib.sites.shortcuts import get_current_site
from django.core.files.storage import default_storage, FileSystemStorage
from django.core.mail import send_mail
from django.utils.timezone import now
from django.core.paginator import Paginator
from django.db.models import Count, Q, Avg, F, Sum
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.translation import gettext as _
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView, ListView, DetailView, CreateView, UpdateView, DeleteView
from django.db import transaction

# Local imports
from .forms import CustomUserCreationForm
from .models import *
from .forms import *


logger = logging.getLogger(__name__)
User = get_user_model()

# --------------------- AUTH VIEWS ---------------------
class CustomLoginView(LoginView):
    template_name = 'login.html'
    redirect_authenticated_user = True

    def form_valid(self, form):
        user = form.get_user()

        if user.is_authenticated:
            if not user.is_auth:
                # Generate OTP and set expiration
                otp = get_random_string(length=6, allowed_chars='1234567890')
                expiry_time = now() + datetime.timedelta(minutes=5)

                # Store in session
                self.request.session['otp_user_id'] = user.id
                self.request.session['otp_code'] = otp
                self.request.session['otp_expires_at'] = expiry_time.isoformat()

                # Send email
                send_mail(
                    'Your OTP Code',
                    f'Your OTP code is: {otp}',
                    'sushilgautam2323@gmail.com',
                    [user.email],
                    fail_silently=False,
                )

                messages.info(self.request, "An OTP was sent to your email.")
                return redirect('verify-otp')

            return super().form_valid(form)

        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('index')

def verify_otp(request):
    if request.method == 'POST':
        otp_input = request.POST.get('otp')
        session = request.session

        user_id = session.get('otp_user_id')
        otp_code = session.get('otp_code')
        expires_at_str = session.get('otp_expires_at')
        try:
            expires_at = datetime.datetime.fromisoformat(expires_at_str)
        except (TypeError, ValueError):
            messages.error(request, "Invalid session. Please login again.")
            return redirect('login')

        if now() > expires_at:
            messages.error(request, "OTP has expired. Please log in again.")
            return redirect('login')

        if otp_input == otp_code:
            try:
                user = User.objects.get(id=user_id)
                user.is_auth = True
                user.save()
                messages.success(request, "OTP verified successfully.")
                # Clean up session
                for key in ['otp_user_id', 'otp_code', 'otp_expires_at']:
                    session.pop(key, None)
                return redirect('account')
            except User.DoesNotExist:
                messages.error(request, "User not found.")
                return redirect('login')

        messages.error(request, "Invalid OTP.")
    
    return render(request, 'verify_otp.html')

class RegisterPage(View):
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('account')
        return render(request, 'register.html', {'form': CustomUserCreationForm()})

    def post(self, request):
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # login(request, user)
            return redirect('login')

        else:
            print("\n--- Form is NOT valid. Errors: ---")
            print(form.errors)
            print("----------------------------------\n")
        return render(request, 'register.html', {'form': form})

class CustomLogoutView(View):
    def get(self, request):
        logout(request)
        return redirect('login')

class PasswordResetRequestView(View):
    def get(self, request):
        return render(request, 'password_reset.html', {'form': PasswordResetForm()})

    def post(self, request):
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            users = User.objects.filter(email=email)
            if users.exists():
                for user in users:
                    subject = "Password Reset Requested"
                    context = {
                        "email": user.email,
                        'domain': get_current_site(request).domain,
                        'site_name': 'WatchHouse',
                        "uid": urlsafe_base64_encode(force_bytes(user.pk)),
                        "token": default_token_generator.make_token(user),
                        "username": user.username,
                    }
                    message = render_to_string("password_reset_email.txt", context)
                    try:
                        send_mail(subject, message, 'your_email@example.com', [user.email], fail_silently=False)
                    except Exception as e:
                        messages.error(request, f"Error sending email: {str(e)}")
                        return render(request, 'password_reset.html', {'form': form})
                messages.success(request, "Password reset email sent successfully!")
                return redirect("login")
            else:
                messages.error(request, "No user is associated with this email address.")
        return render(request, 'password_reset.html', {'form': form})

class PasswordResetConfirmView(View):
    def get(self, request, uidb64, token):
        try:
            # Decode the user ID
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        # Check if user exists and token is valid
        if user is not None and default_token_generator.check_token(user, token):
            # Token is valid, show password reset form
            form = SetPasswordForm(user)
            return render(request, 'password_reset_confirm.html', {
                'form': form,
                'validlink': True,
                'user': user
            })
        else:
            # Invalid token or user
            return render(request, 'password_reset_confirm.html', {
                'form': None,
                'validlink': False
            })

    def post(self, request, uidb64, token):
        try:
            # Decode the user ID
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None

        # Check if user exists and token is valid
        if user is not None and default_token_generator.check_token(user, token):
            form = SetPasswordForm(user, request.POST)
            if form.is_valid():
                # Save the new password
                user = form.save()
                messages.success(request, "Your password has been reset successfully! You can now log in with your new password.")
                
                return redirect('login')
            else:
                # Form has errors, re-render with errors
                return render(request, 'password_reset_confirm.html', {
                    'form': form,
                    'validlink': True,
                    'user': user
                })
        else:
            # Invalid token or user
            return render(request, 'password_reset_confirm.html', {
                'form': None,
                'validlink': False
            })

class IndexView(View):
    def get(self, request):
        return render(request, 'index.html')


# --------------------- TRANSLATE VIEW ---------------------
class TranslateView(LoginRequiredMixin, View):
    TEST_SCRIPTS_DIR = "/media/decoy/myssd/WEB/SLD/SLD_web/test"
    login_url = '/login/'
    
    # File size limits (in bytes)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CAPTURED_IMAGE_SIZE_KB = 5000  # 5MB for base64 image
    MAX_CAPTURED_VIDEO_SIZE_KB = 20000  # 20MB for base64 video
    MAX_UPLOAD_SIZE_MB = 20  # Max file size for uploads
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    ALLOWED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm']

    def get(self, request):
        """Handle GET requests - show initial form"""
        context = {
            'input_type': 'image',
            'category': 'alphabet',
            'prediction': None,
            'media_url': None,
            'confidence': None,
            'processing_time': None,
            'prediction_id': None,
            'feedback_required': False,
            'status': 'ready'
        }
        return render(request, 'translate.html', context)

    def post(self, request):
        """Handle POST requests - process file upload or camera capture and run prediction"""
        input_type = request.POST.get('input_type', 'image').lower()
        category = request.POST.get('category', 'alphabet').lower()
        uploaded_file = request.FILES.get('media')
        captured_image_data = request.POST.get('captured_image', '')
        captured_video_data = request.POST.get('captured_video', '')
        is_captured = request.POST.get('is_captured', 'false') == 'true'

        print(f"[DEBUG] input_type={input_type}, category={category}, is_captured={is_captured}")
        print(f"[DEBUG] uploaded_file={'Yes' if uploaded_file else 'No'}")
        print(f"[DEBUG] captured_image_data_length={len(captured_image_data)}")
        print(f"[DEBUG] captured_video_data_length={len(captured_video_data)}")

        fs = FileSystemStorage()
        file_path = media_url = prediction = raw_prediction = None
        user_media = None
        processing_time = confidence = None
        prediction_obj = None
        status = 'processing'

        # 1. Process captured media
        if is_captured:
            if captured_image_data and input_type == 'image':
                print("[INFO] Processing captured image from base64 data")
                file_path, media_url, user_media = self.process_captured_image(captured_image_data, request.user)
            elif captured_video_data and input_type == 'video':
                print("[INFO] Processing captured video from base64 data")
                file_path, media_url, user_media = self.process_captured_video(captured_video_data, request.user)
            else:
                prediction = f"No captured {input_type} data found"
                status = 'error'
                messages.error(request, prediction)
                print(f"[ERROR] {prediction}")
            
            # Handle capture processing errors
            if isinstance(user_media, str):
                prediction = user_media
                status = 'error'
                user_media = None
                messages.error(request, prediction)
                print(f"[ERROR] {prediction}")

        # 2. Process uploaded file (image or video)
        elif uploaded_file:
            print("[INFO] Processing uploaded file")
            file_type = 'image' if input_type == 'image' else 'video'
            is_valid, message = self.validate_file(uploaded_file, file_type)

            if not is_valid:
                prediction = message
                status = 'error'
                messages.error(request, message)
                print(f"[ERROR] {message}")
            else:
                try:
                    filename = fs.save(uploaded_file.name, uploaded_file)
                    file_path = fs.path(filename)
                    media_url = fs.url(filename)
                    user_media = self.save_user_media(
                        user=request.user,
                        file_path=file_path,
                        media_type=file_type,
                        input_type='upload'
                    )
                    print(f"[SUCCESS] File saved at {file_path}")
                except Exception as e:
                    prediction = f"Error saving file: {e}"
                    status = 'error'
                    messages.error(request, prediction)
                    print(f"[ERROR] {prediction}")

        else:
            # 3. Neither captured nor uploaded
            prediction = f"Please provide or capture a {input_type}."
            status = 'error'
            messages.warning(request, prediction)
            print(f"[WARN] {prediction}")

        # 4. Run prediction if media is ready
        if file_path and not prediction:
            print(f"[INFO] Running prediction on {file_path}")
            script_path = None

            if input_type == 'image':
                if category == 'digit':
                    script_path = os.path.join(self.TEST_SCRIPTS_DIR, 'test_digit.py')
                elif category == 'alphabet':
                    script_path = os.path.join(self.TEST_SCRIPTS_DIR, 'test_alpha.py')
                else:
                    prediction = "Invalid image category."
                    status = 'error'
                    messages.error(request, prediction)

            elif input_type == 'video':
                script_path = os.path.join(self.TEST_SCRIPTS_DIR, 'test_cap.py')

            if script_path and os.path.exists(script_path):
                raw_prediction, processing_time, error = self.run_prediction_script(
                    script_path, file_path, input_type, timeout=60
                )

                if error:
                    prediction = error
                    status = 'error'
                    messages.error(request, prediction)
                    print(f"[ERROR] {prediction}")
                else:
                    prediction, confidence = self.clean_prediction_output(raw_prediction)
                    if prediction and prediction != "No prediction available":
                        status = 'completed'
                        prediction_obj = self.save_prediction(
                            user=request.user,
                            prediction_text=prediction,
                            raw_output=raw_prediction,
                            confidence=confidence,
                            processing_time=processing_time,
                            user_image=user_media if input_type == 'image' else None,
                            user_video=user_media if input_type == 'video' else None
                        )
                        messages.success(request, f"Prediction: {prediction}")
                        print(f"[SUCCESS] Prediction saved: {prediction}")
                    else:
                        prediction = "No prediction available"
                        status = 'failed'
                        messages.warning(request, prediction)
                        print(f"[WARN] Prediction failed")
            elif script_path:
                prediction = f"Prediction script not found: {script_path}"
                status = 'error'
                messages.error(request, prediction)
                print(f"[ERROR] {prediction}")

        # 5. Render template with results
        context = {
            'input_type': input_type,
            'category': category,
            'prediction': prediction,
            'media_url': media_url,
            'confidence': confidence,
            'processing_time': processing_time,
            'prediction_id': prediction_obj.prediction_id if prediction_obj else None,
            'feedback_required': prediction_obj.feedback_required if prediction_obj else False,
            'status': status,
            'has_result': bool(prediction),
            'is_success': status == 'completed',
            'is_error': status == 'error',
            'is_processing': status == 'processing',
        }

        return render(request, 'translate.html', context)

    def validate_file(self, file, file_type):
        """Validate uploaded file size and extension"""
        if not file:
            return False, "No file provided"
        
        # Check file size
        if file_type == 'image' and file.size > self.MAX_IMAGE_SIZE:
            return False, f"Image file too large. Maximum size: {self.MAX_IMAGE_SIZE // (1024*1024)}MB"
        elif file_type == 'video' and file.size > self.MAX_VIDEO_SIZE:
            return False, f"Video file too large. Maximum size: {self.MAX_VIDEO_SIZE // (1024*1024)}MB"
        
        # Check file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_type == 'image' and file_extension not in self.ALLOWED_IMAGE_EXTENSIONS:
            return False, f"Invalid image format. Allowed: {', '.join(self.ALLOWED_IMAGE_EXTENSIONS)}"
        elif file_type == 'video' and file_extension not in self.ALLOWED_VIDEO_EXTENSIONS:
            return False, f"Invalid video format. Allowed: {', '.join(self.ALLOWED_VIDEO_EXTENSIONS)}"
        
        return True, "Valid file"

    def process_captured_image(self, captured_image_data, user):
        """Process base64 captured image data"""
        try:
            if not captured_image_data:
                return None, None, "No captured image data"
            
            # Decode base64 image
            header, encoded = captured_image_data.split(',', 1)
            image_data = base64.b64decode(encoded)
            
            # Validate image data size
            if len(image_data) > self.MAX_IMAGE_SIZE:
                return None, None, "Captured image too large"
            
            # Decode image using OpenCV
            frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return None, None, "Failed to decode captured image"
            
            # Create directory for captured images
            capture_dir = os.path.join(settings.MEDIA_ROOT, 'captured')
            os.makedirs(capture_dir, exist_ok=True)
            
            # Save image
            capture_filename = f'capture_{uuid.uuid4().hex[:8]}.jpg'
            capture_path = os.path.join(capture_dir, capture_filename)
            cv2.imwrite(capture_path, frame)
            
            # Generate media URL
            fs = FileSystemStorage()
            media_url = fs.url(os.path.relpath(capture_path, settings.MEDIA_ROOT))
            
            # Save to database
            user_media = self.save_user_media(
                user=user,
                file_path=capture_path,
                media_type='image',
                input_type='camera'
            )
            
            return capture_path, media_url, user_media
            
        except Exception as e:
            logger.error(f"Captured image processing failed: {e}")
            return None, None, f"Error processing captured image: {str(e)}"

    def process_captured_video(self, captured_video_data, user):
        """Process base64 captured video data"""
        try:
            if not captured_video_data:
                return None, None, "No captured video data"
            
            # Decode base64 video
            header, encoded = captured_video_data.split(',', 1)
            video_data = base64.b64decode(encoded)
            
            # Validate video data size
            if len(video_data) > self.MAX_VIDEO_SIZE:
                return None, None, "Captured video too large"
            
            # Create directory for captured videos
            capture_dir = os.path.join(settings.MEDIA_ROOT, 'captured')
            os.makedirs(capture_dir, exist_ok=True)
            
            # Save video - determine format from header
            if 'webm' in header.lower():
                extension = '.webm'
            elif 'mp4' in header.lower():
                extension = '.mp4'
            else:
                extension = '.webm'  # Default to webm
            
            capture_filename = f'capture_{uuid.uuid4().hex[:8]}{extension}'
            capture_path = os.path.join(capture_dir, capture_filename)
            
            # Write video data directly
            with open(capture_path, 'wb') as f:
                f.write(video_data)
            
            # Verify video file is valid by trying to read it with OpenCV
            cap = cv2.VideoCapture(capture_path)
            if not cap.isOpened():
                cap.release()
                os.remove(capture_path)
                return None, None, "Failed to process captured video - invalid format"
            
            # Get video properties for validation
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            print(f"[DEBUG] Captured video: {frame_count} frames, {fps} fps, {duration:.2f}s duration")
            
            # Generate media URL
            fs = FileSystemStorage()
            media_url = fs.url(os.path.relpath(capture_path, settings.MEDIA_ROOT))
            
            # Save to database
            user_media = self.save_user_media(
                user=user,
                file_path=capture_path,
                media_type='video',
                input_type='camera'
            )
            
            return capture_path, media_url, user_media
            
        except Exception as e:
            logger.error(f"Captured video processing failed: {e}")
            return None, None, f"Error processing captured video: {str(e)}"

    def run_prediction_script(self, script_path, file_path, input_type, timeout=60):
        """Run ML prediction script with timeout"""
        try:
            command = ['python3', script_path, file_path]
            print(f"Executing command: {' '.join(command)}")
            logger.info(f"Subprocess command: {' '.join(command)}")

            start_time = timezone.now()
            
            # Run subprocess with timeout
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                timeout=timeout
            )
            
            end_time = timezone.now()
            processing_time = (end_time - start_time).total_seconds()

            stdout = process.stdout
            stderr = process.stderr

            if stderr:
                logger.warning(f"Script {script_path} stderr: {stderr}")
                print(f"Script stderr: {stderr}")

            logger.info(f"Script finished in {processing_time:.2f}s")
            logger.debug(f"Script stdout: {stdout}")

            return stdout.strip(), processing_time, None
        
        except subprocess.TimeoutExpired as e:
            error_message = f"Script execution timed out after {timeout} seconds."
            logger.error(error_message, exc_info=True)
            return None, None, error_message
            
        except subprocess.CalledProcessError as e:
            error_message = f"Script execution failed with exit code {e.returncode}. Error: {e.stderr.strip() if e.stderr else 'Unknown error'}"
            logger.error(error_message, exc_info=True)
            return None, None, error_message
            
        except FileNotFoundError:
            error_message = f"Python interpreter or script not found: {script_path}"
            logger.error(error_message, exc_info=True)
            return None, None, error_message
            
        except Exception as e:
            error_message = f"Unexpected error during script execution: {str(e)}"
            logger.error(error_message, exc_info=True)
            return None, None, error_message

    def clean_prediction_output(self, raw_output):
        """Clean and extract prediction from raw ML model output"""
        prediction = "No prediction available"
        confidence = None

        if raw_output:
            # Remove TensorFlow warnings and info messages
            lines = [line.strip() for line in raw_output.split('\n') if line.strip()]
            filtered_lines = [
                line for line in lines
                if not line.startswith("W0000") 
                and "inference_feedback_manager.cc" not in line
                and not line.startswith("INFO: Created TensorFlow Lite XNNPACK delegate")
                and not line.startswith("I0000")
            ]
            
            if filtered_lines:
                # Try to find prediction with confidence pattern
                for line in filtered_lines:
                    # Pattern: "Predicted Sign: TEXT (Confidence: XX.XX%)"
                    match = re.search(r"Predicted Sign: (.+?) \(Confidence: (\d+\.?\d*)%\)", line)
                    if match:
                        prediction = match.group(1).strip()
                        try:
                            confidence = float(match.group(2))
                        except ValueError:
                            logger.warning(f"Could not parse confidence: {match.group(2)}")
                        break
                
                # If no confidence pattern found, use last meaningful line
                if prediction == "No prediction available":
                    prediction = filtered_lines[-1]
                    # Try to extract confidence from any line
                    for line in filtered_lines:
                        if "confidence:" in line.lower():
                            try:
                                confidence = float(line.split("confidence:")[-1].strip().rstrip('%'))
                            except:
                                pass
                
                # Clean prediction text
                for prefix in ["Predicted gloss:", "Prediction:", "Result:"]:
                    if prediction.startswith(prefix):
                        prediction = prediction[len(prefix):].strip()
                        break

        return prediction if prediction else "No prediction available", confidence

    def save_user_media(self, user, file_path, media_type, input_type, location=""):
        """Save user media to database"""
        try:
            relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT)
            
            if media_type == 'image':
                user_media, created = UserImageModel.objects.get_or_create(
                    user=user,
                    image=relative_path,
                    defaults={
                        'input_type': input_type,
                        'location': location,
                        'uploaded_at': timezone.now()
                    }
                )
            elif media_type == 'video':
                user_media, created = UserVideoModel.objects.get_or_create(
                    user=user,
                    video=relative_path,
                    defaults={
                        'input_type': input_type,
                        'location': location,
                        'uploaded_at': timezone.now()
                    }
                )
            else:
                logger.error(f"Unsupported media_type: {media_type}")
                return None
            
            if not created:
                user_media.uploaded_at = timezone.now()
                user_media.input_type = input_type
                user_media.location = location
                user_media.save()
            
            logger.info(f"UserMedia saved for {file_path}. Created: {created}")
            return user_media
            
        except Exception as e:
            logger.error(f"Error saving UserMedia for {file_path}: {e}", exc_info=True)
            return None

    def clean_raw_output(self, raw_output):
        """Clean raw output to remove problematic characters for database storage"""
        if not raw_output:
            return ""
        
        try:
            # Remove or replace problematic Unicode characters
            cleaned = raw_output.encode('latin1', errors='ignore').decode('latin1')
            return cleaned
        except UnicodeEncodeError:
            # If that fails, remove all non-ASCII characters
            cleaned = ''.join(char for char in raw_output if ord(char) < 128)
            return cleaned

    def save_prediction(self, user, prediction_text, raw_output="", confidence=None, processing_time=None, user_image=None, user_video=None):
        """Save prediction to database"""
        try:
            # Clean the raw output
            cleaned_raw_output = self.clean_raw_output(raw_output)
            
            prediction = Prediction.objects.create(
                user=user,
                prediction=prediction_text,
                raw_output=cleaned_raw_output,
                confidence_score=confidence,
                processing_time=processing_time,
                image=user_image,
                video=user_video,
                created_at=timezone.now(),
                status='completed' if prediction_text != "No prediction available" else 'failed',
                feedback_required=True
            )
            
            # Log user activity (optional - skip if model doesn't exist)
            try:
                UserActivity.objects.create(
                    user=user,
                    activity_type='prediction',
                    details={
                        'prediction_id': str(prediction.prediction_id),
                        'confidence': confidence,
                        'processing_time': processing_time,
                        'media_type': 'image' if user_image else 'video'
                    }
                )
            except Exception as e:
                logger.warning(f"Could not log user activity: {e}")
            
            # Check for low performance (optional)
            try:
                self.check_low_performance(prediction)
            except Exception as e:
                logger.warning(f"Could not check low performance: {e}")
            
            logger.info(f"Prediction saved for user {user.username}: '{prediction_text}' (ID: {prediction.pk})")
            return prediction
            
        except Exception as e:
            logger.error(f"Error saving prediction for user {user.username}: {e}", exc_info=True)
            return None

    def check_low_performance(self, prediction):
        """Check if prediction has low performance and create report"""
        try:
            reasons = []
            
            # Check confidence score
            if prediction.confidence_score is not None and prediction.confidence_score < 0.7:
                reasons.append('low_confidence')
            
            # Check processing time
            if prediction.processing_time is not None and prediction.processing_time > 30.0:
                reasons.append('long_processing')
            
            # Check if prediction failed
            if prediction.status == 'failed':
                reasons.append('processing_error')
                
        except Exception as e:
            logger.warning(f"Could not create low performance report: {e}")

class FeedbackView(LoginRequiredMixin, View):    
    def get(self, request):
        """Handle GET request - show feedback form if needed"""
        prediction_id = request.GET.get('prediction_id')
        if prediction_id:
            try:
                prediction = get_object_or_404(Prediction, prediction_id=prediction_id, user=request.user)
                context = {
                    'prediction': prediction,
                    'show_feedback_modal': True
                }
                return render(request, 'feedback_form.html', context)
            except:
                messages.error(request, 'Prediction not found')
                return redirect('interactions')
        
        return render(request, 'feedback_form.html')
    
    def post(self, request):
        """Handle POST request - process feedback submission"""
        try:
            # Check if it's an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return self.handle_ajax_request(request)
            
            # Handle regular form submission
            return self.handle_form_submission(request)
            
        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'message': 'Failed to submit feedback'})
            else:
                messages.error(request, 'Failed to submit feedback')
                return redirect('interactions')
    
    def handle_ajax_request(self, request):
        """Handle AJAX feedback submission"""
        try:
            # Parse JSON data if content type is JSON
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                prediction_id = data.get('prediction_id')
                feedback_type = data.get('feedback_type')
                comment = data.get('comment', '')
            else:
                # Handle form data
                prediction_id = request.POST.get('prediction_id')
                feedback_type = request.POST.get('feedback_type')
                comment = request.POST.get('comment', '')
            
            # Validate input
            if not prediction_id or feedback_type not in ['like', 'dislike']:
                return JsonResponse({'success': False, 'message': 'Invalid feedback data'})
            
            # Process feedback
            success, message = self.process_feedback(request.user, prediction_id, feedback_type, comment)
            
            return JsonResponse({
                'success': success,
                'message': message,
                'feedback_type': feedback_type if success else None
            })
            
        except Exception as e:
            logger.error(f"AJAX feedback submission failed: {e}")
            return JsonResponse({'success': False, 'message': 'Failed to submit feedback'})
    
    def handle_form_submission(self, request):
        """Handle regular form submission"""
        try:
            prediction_id = request.POST.get('prediction_id')
            feedback_type = request.POST.get('feedback_type')
            comment = request.POST.get('comment', '')
            
            # Validate input
            if not prediction_id or feedback_type not in ['like', 'dislike']:
                messages.error(request, 'Invalid feedback data')
                return redirect('interactions')
            
            # Process feedback
            success, message = self.process_feedback(request.user, prediction_id, feedback_type, comment)
            
            if success:
                messages.success(request, message)
            else:
                messages.error(request, message)
            
            return redirect('interactions')
            
        except Exception as e:
            logger.error(f"Form feedback submission failed: {e}")
            messages.error(request, 'Failed to submit feedback')
            return redirect('interactions')
    
    def process_feedback(self, user, prediction_id, feedback_type, comment):
        """Process the feedback and update database"""
        try:
            # Get prediction
            prediction = get_object_or_404(Prediction, prediction_id=prediction_id, user=user)
            
            # Update prediction with feedback
            prediction.user_feedback = feedback_type
            prediction.feedback_timestamp = timezone.now()
            prediction.feedback_comment = comment
            prediction.save()
            
            # Log user activity
            UserActivity.objects.create(
                user=user,
                activity_type='feedback',
                details={
                    'prediction_id': str(prediction_id),
                    'feedback_type': feedback_type,
                    'has_comment': bool(comment)
                }
            )
            
            # Update daily metrics
            self.update_daily_metrics(feedback_type)
            
            return True, 'Thank you for your feedback!'
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
            return False, 'Failed to submit feedback'
    
    def update_daily_metrics(self, feedback_type):
        """Update daily metrics with feedback"""
        try:
            today = timezone.now().date()
            metrics, created = SystemMetrics.objects.get_or_create(
                date=today,
                defaults={
                    'total_predictions': 0,
                    'positive_feedback': 0,
                    'negative_feedback': 0,
                }
            )
            
            if feedback_type == 'like':
                metrics.positive_feedback += 1
            else:
                metrics.negative_feedback += 1
            
            # Calculate feedback rate
            total_feedback = metrics.positive_feedback + metrics.negative_feedback
            if metrics.total_predictions > 0:
                metrics.feedback_rate = (total_feedback / metrics.total_predictions) * 100
            
            metrics.save()
            
        except Exception as e:
            logger.error(f"Failed to update daily metrics: {e}")


class HistoryView(LoginRequiredMixin, TemplateView):
    template_name = 'interactions.html'
    login_url = '/login/'
    paginate_by = 6

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        current_user = self.request.user
        search_query = self.request.GET.get('search', '')
        status_filter = self.request.GET.get('status', 'all')
        feedback_filter = self.request.GET.get('feedback', 'all')

        # Get predictions for the current user
        predictions = Prediction.objects.filter(user=current_user)

        # Apply search filter
        if search_query:
            predictions = predictions.filter(prediction__icontains=search_query)

        # Apply status filter
        if status_filter != 'all':
            predictions = predictions.filter(status=status_filter)

        # Apply feedback filter
        if feedback_filter == 'with_feedback':
            predictions = predictions.filter(user_feedback__isnull=False)
        elif feedback_filter == 'without_feedback':
            predictions = predictions.filter(user_feedback__isnull=True)
        elif feedback_filter == 'positive':
            predictions = predictions.filter(user_feedback='like')
        elif feedback_filter == 'negative':
            predictions = predictions.filter(user_feedback='dislike')

        # Prepare history items
        history_items = []
        for pred in predictions:
            media_type = 'Image' if pred.image else 'Video' if pred.video else 'Unknown'
            media_url = pred.image.image.url if pred.image else pred.video.video.url if pred.video else None
            
            history_items.append({
                'id': str(pred.prediction_id),
                'type': 'Prediction',
                'content': pred.prediction,
                'created_at': pred.created_at,
                'media_type': media_type,
                'media_url': media_url,
                'status': pred.status,
                'confidence_score': pred.confidence_score,
                'user_feedback': pred.user_feedback,
                'feedback_given': pred.feedback_given,
                'feedback_pending': pred.feedback_pending,
                'processing_time': pred.processing_time,
                'raw_output': pred.raw_output,
                'feedback_comment': pred.feedback_comment,
                'feedback_timestamp': pred.feedback_timestamp,
            })

        # Sort by creation date (newest first)
        history_items.sort(key=lambda x: x['created_at'], reverse=True)

        # Paginate results
        paginator = Paginator(history_items, self.paginate_by)
        page_obj = paginator.get_page(self.request.GET.get('page'))

        # Calculate statistics
        total_predictions = predictions.count()
        completed_predictions = predictions.filter(status='completed').count()
        failed_predictions = predictions.filter(status='failed').count()
        predictions_with_feedback = predictions.filter(user_feedback__isnull=False).count()
        positive_feedback = predictions.filter(user_feedback='like').count()
        negative_feedback = predictions.filter(user_feedback='dislike').count()

        # Calculate feedback rate
        feedback_rate = (predictions_with_feedback / total_predictions * 100) if total_predictions > 0 else 0

        context.update({
            'history_items': page_obj,
            'search_query': search_query,
            'status_filter': status_filter,
            'feedback_filter': feedback_filter,
            'total_predictions': total_predictions,
            'completed_predictions': completed_predictions,
            'failed_predictions': failed_predictions,
            'predictions_with_feedback': predictions_with_feedback,
            'positive_feedback': positive_feedback,
            'negative_feedback': negative_feedback,
            'feedback_rate': round(feedback_rate, 1),
            'current_user': current_user,
            'status_choices': [
                ('all', 'All Status'),
                ('pending', 'Pending'),
                ('processing', 'Processing'),
                ('completed', 'Completed'),
                ('failed', 'Failed'),
            ],
            'feedback_choices': [
                ('all', 'All Feedback'),
                ('with_feedback', 'With Feedback'),
                ('without_feedback', 'Without Feedback'),
                ('positive', 'Positive'),
                ('negative', 'Negative'),
            ],
        })
        return context

class AccountDashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'account.html'


class ChangeUsernameView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        new_username = request.POST.get('username')
        if new_username:
            request.user.username = new_username
            request.user.save()
            messages.success(request, "Username updated successfully.")
        return redirect('account')


class ChangePasswordView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Password updated successfully.')
        else:
            messages.error(request, 'Please correct the error below.')
        return redirect('account')
    
class ChangeEmailView(LoginRequiredMixin, View):
    def post(self, request):
        new_email = request.POST.get('email')
        if new_email:
            request.user.email = new_email
            request.user.save()
            messages.success(request, "Email updated successfully.")
        else:
            messages.error(request, "Please enter a valid email.")
        return redirect('account_dashboard')
    
class AdminRequiredMixin(UserPassesTestMixin):
    """Mixin to require admin access"""
    def test_func(self):
        return self.request.user.is_authenticated and (
            self.request.user.is_admin or 
            self.request.user.is_superuser or
            hasattr(self.request.user, 'adminprofile')
        )

    def handle_no_permission(self):
        messages.error(self.request, "You don't have permission to access this page.")
        return redirect('index')


class AdminDashboardView(AdminRequiredMixin, TemplateView):
    """Main admin dashboard"""
    template_name = 'admin_dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get date range (last 30 days)
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=30)
        
        # Basic stats
        context['total_users'] = User.objects.count()
        context['active_users'] = User.objects.filter(
            last_login__gte=timezone.now() - timedelta(days=30)
        ).count()
        context['total_predictions'] = Prediction.objects.count()
        context['pending_predictions'] = Prediction.objects.filter(
            status='pending'
        ).count()
        
        # Recent activity
        context['recent_predictions'] = Prediction.objects.select_related(
            'user'
        ).order_by('-created_at')[:5]
        
        context['recent_users'] = User.objects.order_by('-date_joined')[:5]
        
        # Performance metrics
        context['avg_processing_time'] = Prediction.objects.aggregate(
            avg_time=Avg('processing_time')
        )['avg_time'] or 0
        
        context['avg_confidence'] = Prediction.objects.aggregate(
            avg_conf=Avg('confidence_score')
        )['avg_conf'] or 0
        
        # Feedback stats
        total_predictions_with_feedback = Prediction.objects.filter(
            user_feedback__isnull=False
        ).count()
        positive_feedback = Prediction.objects.filter(
            user_feedback='like'
        ).count()
        
        context['feedback_rate'] = (
            (total_predictions_with_feedback / context['total_predictions'] * 100) 
            if context['total_predictions'] > 0 else 0
        )
        
        context['positive_feedback_rate'] = (
            (positive_feedback / total_predictions_with_feedback * 100) 
            if total_predictions_with_feedback > 0 else 0
        )
        
        return context


class AdminUserListView(AdminRequiredMixin, ListView):
    """List all users with admin controls"""
    model = User
    template_name = 'admin_user_list.html'
    context_object_name = 'users'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = User.objects.all().order_by('-date_joined')
        
        # Search functionality
        search_query = self.request.GET.get('search')
        if search_query:
            queryset = queryset.filter(
                Q(username__icontains=search_query) |
                Q(email__icontains=search_query) |
                Q(first_name__icontains=search_query) |
                Q(last_name__icontains=search_query)
            )
        
        # Filter by user type
        user_type = self.request.GET.get('type')
        if user_type == 'admin':
            queryset = queryset.filter(is_admin=True)
        elif user_type == 'active':
            queryset = queryset.filter(is_active=True)
        elif user_type == 'inactive':
            queryset = queryset.filter(is_active=False)
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['search_query'] = self.request.GET.get('search', '')
        context['user_type'] = self.request.GET.get('type', '')
        return context


class AdminUserDetailView(AdminRequiredMixin, DetailView):
    """Detailed view of a specific user"""
    model = User
    template_name = 'admin_user_detail.html'
    context_object_name = 'user_obj'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.get_object()
        
        # User stats
        context['user_predictions'] = Prediction.objects.filter(
            user=user
        ).count()
        
        context['user_images'] = UserImageModel.objects.filter(
            user=user
        ).count()
        
        context['user_videos'] = UserVideoModel.objects.filter(
            user=user
        ).count()
        
        # Recent activity
        context['recent_predictions'] = Prediction.objects.filter(
            user=user
        ).order_by('-created_at')[:5]
        
        context['recent_activity'] = UserActivity.objects.filter(
            user=user
        ).order_by('-timestamp')[:10]
        
        # Performance metrics
        user_predictions = Prediction.objects.filter(user=user)
        context['avg_confidence'] = user_predictions.aggregate(
            avg_conf=Avg('confidence_score')
        )['avg_conf'] or 0
        
        context['positive_feedback_count'] = user_predictions.filter(
            user_feedback='like'
        ).count()
        
        context['negative_feedback_count'] = user_predictions.filter(
            user_feedback='dislike'
        ).count()
        
        return context


class AdminUserCreateView(AdminRequiredMixin, SuccessMessageMixin, CreateView):
    """Create new admin user"""
    model = User
    form_class = CustomUserCreationForm
    template_name = 'admin_user_create.html'
    success_url = reverse_lazy('user_list')
    success_message = "Admin user created successfully!"
    
    def form_valid(self, form):
        response = super().form_valid(form)
        # Log activity
        UserActivity.objects.create(
            user=self.request.user,
            activity_type='admin_user_created',
            details={
                'created_user': self.object.username,
                'is_admin': form.cleaned_data.get('is_admin', False)
            }
        )
        
        return response


class AdminUserUpdateView(AdminRequiredMixin, SuccessMessageMixin, UpdateView):
    """Update user information"""
    model = User
    form_class = UserUpdateForm
    template_name = 'admin_user_update.html'
    success_message = "User updated successfully!"
    
    def get_success_url(self):
        return reverse_lazy('admin:user_detail', kwargs={'pk': self.object.pk})

class AdminUserDeleteView(AdminRequiredMixin, SuccessMessageMixin, DeleteView):
    model = User
    template_name = 'admin_user_confirm_delete.html'
    context_object_name = 'user_obj'
    success_url = reverse_lazy('user_list')
    success_message = "User deleted successfully!"

    def delete(self, request, *args, **kwargs):
        self.object = self.get_object()

        # Log activity before deletion
        UserActivity.objects.create(
            user=request.user,
            activity_type='admin_user_deleted',
            details={
                'deleted_user': self.object.username,
                'user_id': self.object.id
            }
        )

        messages.success(self.request, self.success_message)
        return super().delete(request, *args, **kwargs)


class AdminPredictionListView(AdminRequiredMixin, ListView):
    """List all predictions with admin controls"""
    model = Prediction
    template_name = 'admin_prediction_list.html'
    context_object_name = 'predictions'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Prediction.objects.select_related('user').order_by('-created_at')
        
        # Filter by status
        status = self.request.GET.get('status')
        if status:
            queryset = queryset.filter(status=status)
        
        # Filter by feedback
        feedback = self.request.GET.get('feedback')
        if feedback == 'positive':
            queryset = queryset.filter(user_feedback='like')
        elif feedback == 'negative':
            queryset = queryset.filter(user_feedback='dislike')
        elif feedback == 'pending':
            queryset = queryset.filter(user_feedback__isnull=True)
            
        # Search by user
        user_search = self.request.GET.get('user')
        if user_search:
            queryset = queryset.filter(user__username__icontains=user_search)
            
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['status_filter'] = self.request.GET.get('status', '')
        context['feedback_filter'] = self.request.GET.get('feedback', '')
        context['user_search'] = self.request.GET.get('user', '')
        return context

def is_admin(user):
    return user.is_admin

@login_required
@user_passes_test(is_admin)
def admin_prediction_delete(request, pk):
    prediction = get_object_or_404(Prediction, prediction_id=pk)
    
    if request.method == 'POST':
        delete_media = request.POST.get('delete_media', False)
        
        try:
            with transaction.atomic():
                # Get related objects before deletion
                related_image = prediction.image
                related_video = prediction.video
                
                # Delete the prediction (this will cascade to PredictionFeedback)
                prediction.delete()
                
                # Optionally delete media files if they're not referenced by other predictions
                if delete_media:
                    if related_image and not Prediction.objects.filter(image=related_image).exists():
                        related_image.delete()
                    if related_video and not Prediction.objects.filter(video=related_video).exists():
                        related_video.delete()
                
                messages.success(request, 'Prediction deleted successfully.')
                return redirect('prediction_list')
                
        except Exception as e:
            messages.error(request, f'Error deleting prediction: {str(e)}')
            return redirect('prediction_detail', pk=pk)
    
    # GET request - show confirmation page with related objects info
    related_objects = get_related_objects_info(prediction)
    
    return render(request, 'prediction_delete.html', {
        'prediction': prediction,
        'related_objects': related_objects
    })

def get_related_objects_info(prediction):
    """Get information about objects that will be affected by deletion"""
    related_info = {
        'feedback': None,
        'image': None,
        'video': None,
        'image_used_elsewhere': False,
        'video_used_elsewhere': False,
    }
    
    # Check for detailed feedback
    if hasattr(prediction, 'detailed_feedback'):
        related_info['feedback'] = prediction.detailed_feedback
    
    # Check for related media and if it's used elsewhere
    if prediction.image:
        related_info['image'] = prediction.image
        related_info['image_used_elsewhere'] = Prediction.objects.filter(
            image=prediction.image
        ).exclude(prediction_id=prediction.prediction_id).exists()
    
    if prediction.video:
        related_info['video'] = prediction.video
        related_info['video_used_elsewhere'] = Prediction.objects.filter(
            video=prediction.video
        ).exclude(prediction_id=prediction.prediction_id).exists()
    
    return related_info

@login_required
@user_passes_test(is_admin)
def admin_prediction_detail(request, pk):
    # Filter by prediction_id (not pk) to match your URL config and templates
    prediction = get_object_or_404(
        Prediction.objects.select_related('user', 'image', 'video')
        .prefetch_related('detailed_feedback'), 
        prediction_id=pk
    )
    
    # Get related objects info for display
    related_objects = get_related_objects_info(prediction)
    
    return render(request, 'prediction_detail.html', {
        'prediction': prediction,
        'related_objects': related_objects
    })


class AdminSystemMetricsView(AdminRequiredMixin, TemplateView):
    template_name = 'admin_metrics.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Get date range from query parameters
        days = int(self.request.GET.get('days', 30))
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)

        # Get daily metrics from SystemMetrics model
        daily_metrics = SystemMetrics.objects.filter(
            date__gte=start_date,
            date__lte=end_date
        ).order_by('date')

        context['daily_metrics'] = daily_metrics
        context['days'] = days

        # Summary statistics
        summary_stats = self.get_summary_stats(start_date, end_date)
        context['summary_stats'] = summary_stats

        # Chart data preparation
        context['chart_data'] = self.prepare_chart_data(start_date, days)

        # Recent activity
        context['recent_activity'] = self.get_recent_activity()

        # Top performing predictions
        context['top_predictions'] = self.get_top_predictions()

        return context

    def get_summary_stats(self, start_date, end_date):
        """Calculate summary statistics for the date range"""
        # Total predictions in date range
        total_predictions = Prediction.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date
        ).count()

        # Successful predictions (with positive feedback)
        successful_predictions = Prediction.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date,
            user_feedback='like'
        ).count()

        # Failed predictions (with negative feedback)
        failed_predictions = Prediction.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date,
            user_feedback='dislike'
        ).count()

        # Average processing time and confidence
        avg_stats = Prediction.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date
        ).aggregate(
            avg_processing_time=Avg('processing_time'),
            avg_confidence=Avg('confidence_score')
        )

        # New users in date range
        new_users = CustomUser.objects.filter(
            date_joined__date__gte=start_date,
            date_joined__date__lte=end_date
        ).count()

        # Active users (users who made predictions)
        active_users = CustomUser.objects.filter(
            prediction__created_at__date__gte=start_date,
            prediction__created_at__date__lte=end_date
        ).distinct().count()

        # Feedback rate
        predictions_with_feedback = Prediction.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date,
            user_feedback__isnull=False
        ).count()

        feedback_rate = (predictions_with_feedback / total_predictions * 100) if total_predictions > 0 else 0

        return {
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions,
            'failed_predictions': failed_predictions,
            'avg_processing_time': round(avg_stats['avg_processing_time'] or 0, 2),
            'avg_confidence': round(avg_stats['avg_confidence'] or 0, 2),
            'new_users': new_users,
            'active_users': active_users,
            'feedback_rate': round(feedback_rate, 2),
            'success_rate': round((successful_predictions / total_predictions * 100) if total_predictions > 0 else 0, 2)
        }

    def prepare_chart_data(self, start_date, days):
        """Prepare data for charts"""
        predictions_data = []
        users_data = []
        feedback_data = []
        performance_data = []

        for i in range(days):
            date = start_date + timedelta(days=i)
            
            # Daily predictions
            daily_predictions = Prediction.objects.filter(created_at__date=date)
            predictions_count = daily_predictions.count()
            
            # Daily user registrations
            users_count = CustomUser.objects.filter(date_joined__date=date).count()
            
            # Daily feedback counts
            positive_feedback = daily_predictions.filter(user_feedback='like').count()
            negative_feedback = daily_predictions.filter(user_feedback='dislike').count()
            pending_feedback = daily_predictions.filter(user_feedback__isnull=True).count()
            
            # Daily performance metrics
            avg_processing_time = daily_predictions.aggregate(
                avg=Avg('processing_time')
            )['avg'] or 0
            
            avg_confidence = daily_predictions.aggregate(
                avg=Avg('confidence_score')
            )['avg'] or 0

            predictions_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'label': date.strftime('%b %d'),
                'count': predictions_count
            })
            
            users_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'label': date.strftime('%b %d'),
                'count': users_count
            })
            
            feedback_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'label': date.strftime('%b %d'),
                'positive': positive_feedback,
                'negative': negative_feedback,
                'pending': pending_feedback
            })
            
            performance_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'label': date.strftime('%b %d'),
                'processing_time': round(avg_processing_time, 2),
                'confidence': round(avg_confidence, 2)
            })

        # Overall feedback distribution
        overall_feedback = {
            'positive': Prediction.objects.filter(user_feedback='like').count(),
            'negative': Prediction.objects.filter(user_feedback='dislike').count(),
            'pending': Prediction.objects.filter(user_feedback__isnull=True).count(),
        }

        return {
            'predictions': predictions_data,
            'users': users_data,
            'feedback_daily': feedback_data,
            'feedback_overall': overall_feedback,
            'performance': performance_data,
        }

    def get_recent_activity(self):
        """Get recent user activity"""
        return UserActivity.objects.select_related('user').order_by('-timestamp')[:10]

    def get_top_predictions(self):
        """Get top predictions by confidence score"""
        return Prediction.objects.select_related('user').filter(
            confidence_score__isnull=False
        ).order_by('-confidence_score')[:10]


class AdminMetricsAPIView(AdminRequiredMixin, TemplateView):
    """API endpoint for real-time analytics data"""
    
    def get(self, request, *args, **kwargs):
        # Get date range
        days = int(request.GET.get('days', 30))
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Predictions over time
        predictions_data = []
        feedback_data = []
        
        for i in range(days):
            date = start_date + timedelta(days=i)
            daily_predictions = Prediction.objects.filter(created_at__date=date)
            
            predictions_count = daily_predictions.count()
            positive_feedback = daily_predictions.filter(user_feedback='like').count()
            negative_feedback = daily_predictions.filter(user_feedback='dislike').count()
            pending_feedback = daily_predictions.filter(user_feedback__isnull=True).count()
            
            predictions_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': predictions_count
            })
            
            feedback_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'positive': positive_feedback,
                'negative': negative_feedback,
                'pending': pending_feedback
            })
        
        # User registration over time
        users_data = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            count = CustomUser.objects.filter(
                date_joined__date=date
            ).count()
            users_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': count
            })
        
        # Overall feedback distribution
        overall_feedback = {
            'positive': Prediction.objects.filter(user_feedback='like').count(),
            'negative': Prediction.objects.filter(user_feedback='dislike').count(),
            'pending': Prediction.objects.filter(user_feedback__isnull=True).count(),
        }
        
        # Performance metrics
        performance_data = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            daily_predictions = Prediction.objects.filter(created_at__date=date)
            
            avg_processing_time = daily_predictions.aggregate(
                avg=Avg('processing_time')
            )['avg'] or 0
            
            avg_confidence = daily_predictions.aggregate(
                avg=Avg('confidence_score')
            )['avg'] or 0
            
            performance_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'processing_time': round(avg_processing_time, 2),
                'confidence': round(avg_confidence, 2)
            })
        
        # Summary statistics
        total_predictions = Prediction.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date
        ).count()
        
        successful_predictions = Prediction.objects.filter(
            created_at__date__gte=start_date,
            created_at__date__lte=end_date,
            user_feedback='like'
        ).count()
        
        success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        return JsonResponse({
            'predictions': predictions_data,
            'users': users_data,
            'feedback_daily': feedback_data,
            'feedback_overall': overall_feedback,
            'performance': performance_data,
            'summary': {
                'total_predictions': total_predictions,
                'successful_predictions': successful_predictions,
                'success_rate': round(success_rate, 2),
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'days': days
                }
            }
        })


