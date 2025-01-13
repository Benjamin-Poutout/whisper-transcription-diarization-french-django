import tempfile
import os
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from pydub import AudioSegment
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import soundfile as sf
from pyannote.audio import Pipeline
import json
import time

diarization_pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization",
  use_auth_token="your_token_here")
diarization_pipeline = diarization_pipeline.to(torch.device('cuda:0'))

# Modèle ASR (chargé au démarrage)
model_id = "bofenghuang/whisper-large-v3-french"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
model_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

class TranscriptionConsumer(AsyncWebsocketConsumer):

    def __init__(self):
        self.audio_chunks = []  # Liste des morceaux audio
        self.recording_stopped = False
        self.groups = {}

    async def connect(self):
        if self.groups is None:
            self.groups = {}
        await self.accept()
        logging.info("WebSocket connection established.")

    async def disconnect(self, close_code):
        logging.info("WebSocket connection closed.")

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Sauvegarde temporaire des chunks audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                tmp_file.write(bytes_data)
                tmp_file_path = tmp_file.name
            
            # Convertir et transcrire
            wav_path = tmp_file_path.replace('.webm', '.wav')
            self.convert_webm_to_wav(tmp_file_path, wav_path)
            transcription = self.transcribe_audio(wav_path)

            transcription_message = json.dumps({
                "type": "transcription",
                "data": transcription
            })
            await self.send(text_data=transcription_message)

            if self.is_recording_stopped():
                print(f"Calling diarization on {wav_path}")
                num_speakers = self.groups.get('numSpeakers', None)  # Récupérer le nombre de locuteurs
                diarization_results = await self.perform_diarization(wav_path)
                print(f"Diarization results: {diarization_results}")  # Log de la diarisation
                diarization_message = json.dumps({
                    "type": "diarization",
                    "data": diarization_results
                })
                await self.send(text_data=diarization_message)


            # Supprimer les fichiers temporaires
            os.remove(tmp_file_path)
            os.remove(wav_path)

        
        elif text_data:
            data = json.loads(text_data)
            if data.get('message') == 'stop':
                # Si le message 'stop' est reçu, marquer l'enregistrement comme terminé
                self.recording_stopped = True
                print("Recording stopped set to True")  # Vérification de l'état
                if self.is_recording_stopped():
                    print("Performing diarization after stop")

            elif data.get('type') == 'set_speakers':
                num_speakers = data.get('numSpeakers')
                self.groups['numSpeakers'] = num_speakers
                print(f"Number of speakers set to: {num_speakers}")

    def convert_webm_to_wav(self, input_path, output_path):
        try:
            audio = AudioSegment.from_file(input_path, format="webm")
            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            audio.export(output_path, format="wav")
        except Exception as e:
            logging.error(f"Error converting WebM to WAV: {e}")

    def transcribe_audio(self, file_path, start_time=None, end_time=None, return_timestamps=True):
        try:
            # Lire le fichier audio
            audio_data, sample_rate = sf.read(file_path)
            
            # Vérifier si le taux d'échantillonnage est 16000
            if sample_rate != 16000:
                raise ValueError(f"Expected sample rate: 16000, but got: {sample_rate}")
            
            # Si start_time et end_time sont fournis, découper l'audio
            if start_time is not None and end_time is not None:
                # Découper l'audio en fonction des timestamps
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                audio_data = audio_data[start_sample:end_sample]
            
            # Traiter l'audio avec le modèle pipeline
            result = model_pipeline(audio_data, return_timestamps=return_timestamps)
            
            # Si le modèle retourne du texte (sans segments temporels)
            if "text" in result:
                return result["text"]
            else:
                logging.error("Unexpected result format: 'text' not found.")
                return "Error: Unexpected result format."
    
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return f"Error during transcription: {str(e)}"
        
    def is_recording_stopped(self):
        # Vérifier si l'enregistrement a été arrêté (basé sur le signal 'stop')
        return self.recording_stopped
        
    async def process_full_audio(self, bytes_data=None, text_data=None):

        if self.is_recording_stopped():
            # Sauvegarde temporaire des chunks audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
                tmp_file.write(bytes_data)
                tmp_file_path = tmp_file.name
            
            # Convertir et transcrire
            wav_path = tmp_file_path.replace('.webm', '.wav')
            self.convert_webm_to_wav(tmp_file_path, wav_path)
            diarization = self.perform_diarization(wav_path)

            # Supprimer les fichiers temporaires
            os.remove(tmp_file_path)
            os.remove(wav_path)

            await self.send(text_data=diarization)

    async def perform_diarization(self, wav_path, num_speakers=None):
        if num_speakers is None:
            num_speakers = self.groups.get('numSpeakers', None)

        try:
            print(f"Starting diarization on file: {wav_path}")
            start_time = time.time()
            diarization_results = diarization_pipeline(wav_path, num_speakers=num_speakers)
            end_time = time.time()
            diarization_duration = end_time - start_time  # Durée en secondes
            print(f"Diarization completed in {diarization_duration:.2f} seconds.")
            print("Num Speakers:", num_speakers)
            print("Diarization results:", diarization_results)

            # Liste pour stocker les chunks à transcrire
            chunks_to_transcribe = []

            # Diviser les segments de diarisation en chunks
            for turn, _, speaker in diarization_results.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                speaker_label = speaker
                
                # Créer un chunk pour chaque segment du locuteur
                chunk = {
                    'start': start,
                    'end': end,
                    'speaker': speaker_label,
                    'wav_path': wav_path
                }
                chunks_to_transcribe.append(chunk)

            # Effectuer la transcription pour chaque chunk
            transcription_results = []
            for chunk in chunks_to_transcribe:
                # Transcrire chaque segment audio
                transcription = self.transcribe_audio(chunk['wav_path'], start_time=chunk['start'], end_time=chunk['end'], return_timestamps=True)
                
                # Ajouter le résultat de transcription à la liste
                transcription_results.append({
                    'speaker': chunk['speaker'],
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'transcription': transcription
                })

            # Affichage des résultats :
            diarization_info = []
            for result in transcription_results:
                diarization_info.append(f"{result['speaker']}: {result['transcription']}")

            return "\n".join(diarization_info)

        except Exception as e:
            logging.error(f"Error during diarization: {e}")
            return f"Error during diarization: {str(e)}"
