from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tempfile
import os
from .consumers import TranscriptionConsumer

def index(request):
    return render(request, 'index.html')


def transcription(request):
    # Votre logique pour gérer la transcription
    return JsonResponse({"status": "success", "message": "Transcription completed"})