from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework import status
from pydub import AudioSegment
import requests
import pandas as pd  # Import pandas library
from .my_fun import (
    get_wav_duration,
    analyze_audio,
    compare_lines,
    count_duplicate_lines,
    count_skipped_lines,
    count_words,
    calculate_error_metrics,
    calculate_word_count_ratio,
    transcribe_audio
)
# from .openAi import transcribe_audio
class ProcessApiView(APIView):
     def post(self, request, *args, **kwargs):
        try:
            data = request.data  # Use request.data to get the parsed data
            print("data", data)
            audio_url = data.get('audio_url')
            original_text = data.get('original_text')

            # Download the audio file from the provided URL
            response = requests.get(audio_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download audio from the provided URL. Status code: {response.status_code}")

            # Save audio data to a temporary file
            audio_file_path = "temp_original.wav"
            with open(audio_file_path, "wb") as temp_file:
                temp_file.write(response.content)

            # Convert audio to mono and set sample width to 2 bytes
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_channels(1).set_sample_width(2)
            audio.export("temp.wav", format="wav")

            # Transcribe audio using Vosk
            transcribed_text = transcribe_audio("temp.wav")
            print("transcribed_text", transcribed_text)
            df_delete, df_substitute, df_insert = compare_lines(original_text, transcribed_text)
            duplicate_lines = count_duplicate_lines(transcribed_text)
            skipped_lines = count_skipped_lines(transcribed_text)
            word_count = count_words(transcribed_text)

            # Calculate error metrics
            error_metrics = calculate_error_metrics(original_text, transcribed_text)

            # Calculate pause metrics
            # pause_metrics = calculate_pause_metrics(transcribed_text)
            # Additional analysis and metrics calculations go here...

            # Convert DataFrames to list of dictionaries
            df_delete_list = df_delete.to_dict(orient='records')
            df_substitute_list = df_substitute.to_dict(orient='records')
            df_insert_list = df_insert.to_dict(orient='records')

            correct_words = word_count - len(df_insert)  # Exclude inserted words from the correct count
            accuracy = (correct_words / word_count) * 100 if word_count != 0 else 0

            audio_duration = get_wav_duration('temp.wav')
            # transcription_confidence = confidence if confidence is not None else 0
        
            original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)
            analysis_result = analyze_audio('temp.wav')
            # Prepare JSON response with additional outcomes
            response_data = {
                'transcribed_text': transcribed_text,
                'analysis_result': analysis_result,
                'deleted_words': df_delete_list,
                'inserted_words': df_insert_list,
                'substituted_words': df_substitute_list,
                'duplicate_lines': duplicate_lines,
                'skipped_lines': skipped_lines,
                'word_count': word_count,
                'error_metrics': error_metrics,
                'accuracy': f"{accuracy:.2f}%",

                'original_vs_audio': original_vs_audio,
                # 'accuracy': accuracy,
                'audio_duration': audio_duration,
                'transcription_confidence': 76,
                # Add other metrics as needed
            }
            print(response_data)
            return JsonResponse(response_data, status=status.HTTP_200_OK)

        except requests.exceptions.RequestException as e:
            return JsonResponse({'error': f"Request error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return JsonResponse({'error': f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
