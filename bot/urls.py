from django.conf.urls import url
from django.urls import path
from bot import views

#Template Taging
app_name = 'bot'
urlpatterns = [
    url(r'^$',views.auth,name='login'),
    path('login',views.login,name='token'),
    path('results',views.results, name='results'),
    path('balance',views.account_balance,name='balance'),
    path('assistant',views.assistant,name='assistant'),
    path('api', views.apiCall, name='apiCall')
]
