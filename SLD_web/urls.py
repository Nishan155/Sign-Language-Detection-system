from django.urls import path
from . import views 
from django.contrib.auth.views import PasswordResetDoneView
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Authentication
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('register/', views.RegisterPage.as_view(), name='Register'),
    path('password-reset/', views.PasswordResetRequestView.as_view(), name='password_reset'),
    path('password_reset_done/', PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('logout/', views.CustomLogoutView.as_view(), name='logout'),
    path('password-reset-confirm/<uidb64>/<token>/', views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('password-reset-complete/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),


    path('', views.IndexView.as_view(), name='index'),
    path('interactions/', views.HistoryView.as_view(), name='interactions'),
    path('translator/', views.TranslateView.as_view(), name='translator'),

    path('dashboard/', views.AccountDashboardView.as_view(), name='account'),
    path('change-username/', views.ChangeUsernameView.as_view(), name='change_username'),
    path('change-password/', views.ChangePasswordView.as_view(), name='change_password'),
    path('account/change-email/', views.ChangeEmailView.as_view(), name='change_email'),
    path('verify-otp/', views.verify_otp, name = "verify-otp"),
    path('feedback',views.FeedbackView.as_view(), name="feedback"),

    path('admins', views.AdminDashboardView.as_view(), name='dashboard'),
    path('users/', views.AdminUserListView.as_view(), name='user_list'),
    path('users/<int:pk>/', views.AdminUserDetailView.as_view(), name='user_detail'),
    path('users/create/', views.AdminUserCreateView.as_view(), name='user_create'),
    path('users/<int:pk>/update/', views.AdminUserUpdateView.as_view(), name='user_update'),
    path('predictions/', views.AdminPredictionListView.as_view(), name='prediction_list'),
    path('predictions/delete/<str:pk>/', views.admin_prediction_delete, name='admin_prediction_delete'),
    path('admin/predictions/<str:pk>/detail/', views.admin_prediction_detail, name='admin_prediction_detail'),
    path('metrics/', views.AdminSystemMetricsView.as_view(), name='metrics'),
    path('users/<int:pk>/delete/', views.AdminUserDeleteView.as_view(), name='user_delete'),
    path('api/analytics/', views.AdminMetricsAPIView.as_view(), name='analytics_api'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)