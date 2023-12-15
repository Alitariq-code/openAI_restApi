from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework import status
from pydub import AudioSegment
import requests
from rest_framework.response import Response 
import pandas as pd  # Import pandas library
from asgiref.sync import async_to_sync
import logging
import json
from .my_fun import (
    get_wav_duration,
    analyze_audio,
    compare_lines,
    count_duplicate_lines,
    count_skipped_lines,
    count_words,
    calculate_error_metrics,
    calculate_word_count_ratio,
    # transcribe_audio
)
# from .backend import transcribe_audio
import timeit
import whisper
import asyncio
model = None    

async def load_model():
    global model
    # Load the model asynchronously
    model = await asyncio.to_thread(whisper.load_model, "base")

async def transcribe_audio(file_path):
    print("doinggggg")
    if model is None:
        await load_model()
    # Transcribe the audio file
    result = await asyncio.to_thread(model.transcribe, file_path)
    print(result["text"])
    return result["text"]
    
def calculate_pause_metrics(transcribed_text):
    # Your logic for pause metrics calculation goes here
    # Placeholder values are used, replace them with actual calculations
    pauses_1_3_seconds = 5
    hesitations_3_seconds = 2

    pause_metrics = {
        'Pauses (1-3 seconds)': pauses_1_3_seconds,
        'Hesitations (3+ seconds)': hesitations_3_seconds,
        # Add other pause metrics as needed
    }

    return pause_metrics
def format_word_list(word_list):
    return [{'ID': entry['ID'], 'Word': entry['Word']} for entry in word_list]










bufferData=[]

class ProcessApiView(APIView):
   
    @async_to_sync
    async def post(self, request, *args, **kwargs):
        try:
            data = request.data  # Use request.data to get the parsed data
            audio_url = data.get('audio_url')
            original_text = data.get('original_text')
            id = data.get('id')
            transcrib = ''
            print(id)

            # Download the audio file from the provided URL
            if id and audio_url:
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

                transcribed_text = await transcribe_audio("temp.wav")
                newData = {}
                newData["id"] = id
                newData["Text"] = transcribed_text
                bufferData.append(newData)
                print(bufferData)
                response = {
            'staus': 'done with audio at openAi'
        }
                return Response(response, status=status.HTTP_200_OK)
            else:
                print("okok")
                print("Searching for ID:", id)
                id_found = False
                while True:
                    for item in bufferData:
                        print("Checking item:", item)
                        if 'id' in item and item['id'] == id:
                            transcrib = item['Text']
                            print("Data of this:", transcrib)
                            id_found = True  # Set the flag to True when ID is found
                            break

                    if id_found:
                        break  # Break out of the outer while loop when ID is found
                substituted_words, delete_words, insert_words,merged = compare_lines(original_text, transcrib)
                duplicate_lines = count_duplicate_lines(transcrib)
                skipped_lines = count_skipped_lines(transcrib)
                word_count = count_words(transcrib)

                correct_words = word_count - len(insert_words)  # Exclude inserted words from the correct count

                audio_duration = get_wav_duration('temp.wav')
                pause_metrics = calculate_pause_metrics(transcrib)

                # Convert DataFrames to lists of dictionaries
                delete, insert, sub = len(delete_words), len(insert_words), len(substituted_words)
                print(delete, insert, sub)
                error_metrics = calculate_error_metrics(original_text, transcrib, delete=delete, insert=insert, sub=sub)

                formatted_deleted_words = format_word_list(delete_words)
                formatted_inserted_words = format_word_list(insert_words)
                formatted_substituted_words = format_word_list(substituted_words)
                merged_foramted = format_word_list(merged)

                accuracy = error_metrics['Acc']
                original_vs_audio = calculate_word_count_ratio(transcrib, original_text)

                analysis_result = analyze_audio('temp.wav')
                response_data = {
                    'transcribed_text': transcrib,
                    'analysis_result': analysis_result,
                    'deleted_words': formatted_deleted_words,
                    'inserted_words': formatted_inserted_words,
                    'substituted_words': formatted_substituted_words,
                    'duplicate_lines': duplicate_lines,
                    'skipped_lines': skipped_lines,
                    'merged':merged,
                    'word_count': word_count,
                    'error_metrics': error_metrics,
                    'accuracy': accuracy,
                    'original_vs_audio': original_vs_audio,
                    'audio_duration': audio_duration,
                    'transcription_confidence': 76,
                    # Add other metrics as needed
                }
                print(response_data)
                return Response(response_data, status=status.HTTP_200_OK)


        except requests.exceptions.RequestException as e:
            return Response({'error': f"Request error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({'error': f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    async def async_response(self, data):
        return Response(data, status=status.HTTP_200_OK)



# class ProcessApiView(APIView):
#     @async_to_sync
#     async def post(self, request, *args, **kwargs):    
#         # print("View is executing...")
#         try:
#             data = request.data  # Use request.data to get the parsed data
#             # print("data", data)
#             audio_url = data.get('audio_url')
#             original_text = data.get('original_text')
#             id = data.get('id')
#             print(id)

            
#             # Download the audio file from the provided URL
#             response = requests.get(audio_url)
#             if response.status_code != 200:
#                 raise Exception(f"Failed to download audio from the provided URL. Status code: {response.status_code}")

#             # Save audio data to a temporary file
#             audio_file_path = "temp_original.wav"
#             with open(audio_file_path, "wb") as temp_file:
#                 temp_file.write(response.content)
#             # print('hello')
#             # Convert audio to mono and set sample width to 2 bytes
#             audio = AudioSegment.from_file(audio_file_path)
#             audio = audio.set_channels(1).set_sample_width(2)
#             audio.export("temp.wav", format="wav")

#             # print("before")            
#             transcribed_text = await transcribe_audio("temp.wav")

#             # print("after")
#             # print("transcribed_text", transcribed_text)
#             substituted_words,delete_words,insert_words= compare_lines(original_text, transcribed_text)
#             duplicate_lines = count_duplicate_lines(transcribed_text)
#             skipped_lines = count_skipped_lines(transcribed_text)
#             word_count = count_words(transcribed_text)

#             # Calculate error metrics
#             # error_metrics = calculate_error_metrics(original_text, transcribed_text)

#             # Calculate pause metrics
#             # pause_metrics = calculate_pause_metrics(transcribed_text)
#             # Additional analysis and metrics calculations go here...

#             # Convert DataFrames to list of dictionaries
#             # df_delete_list = df_delete.to_dict(orient='records')
#             # df_substitute_list = df_substitute.to_dict(orient='records')
#             # df_insert_list = df_insert.to_dict(orient='records')

#             correct_words = word_count - len(insert_words)  # Exclude inserted words from the correct count
#             # accuracy = (correct_words / word_count) * 100 if word_count != 0 else 0

#             audio_duration = get_wav_duration('temp.wav')
#            # Calculate pause metrics
#             pause_metrics = calculate_pause_metrics(transcribed_text)

#         # Convert DataFrames to lists of dictionaries
#             delete, insert, sub = len(delete_words), len(insert_words), len(substituted_words)
#             print(delete,insert,sub)
#             error_metrics = calculate_error_metrics(original_text, transcribed_text, delete=delete, insert=insert, sub=sub)

#             formatted_deleted_words = format_word_list(delete_words)
#             formatted_inserted_words = format_word_list(insert_words)
#             formatted_substituted_words = format_word_list(substituted_words)



       
#             accuracy=error_metrics['Acc']

#             original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)
        
#             # original_vs_audio = calculate_word_count_ratio(transcribed_text, original_text)
#             analysis_result = analyze_audio('temp.wav')
#             # Prepare JSON response with additional outcomes
#             response_data = {
#                 'transcribed_text': transcribed_text,
#                 'analysis_result': analysis_result,
#                 'deleted_words': formatted_deleted_words,
#                 'inserted_words': formatted_inserted_words,
#                 'substituted_words': formatted_substituted_words,
#                 'duplicate_lines': duplicate_lines,
#                 'skipped_lines': skipped_lines,
#                 'word_count': word_count,
#                 'error_metrics': error_metrics,
#                 'accuracy': accuracy,

#                 'original_vs_audio': original_vs_audio,
#                 # 'accuracy': accuracy,
#                 'audio_duration': audio_duration,
#                 'transcription_confidence': 76,
#                 # Add other metrics as needed
#             }
#             print(response_data)
#             return Response(response_data, status=status.HTTP_200_OK)

#         except requests.exceptions.RequestException as e:
#              return Response({'error': f"Request error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#         except Exception as e:
#              return Response({'error': f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
#     async def async_response(self, data):
#         return Response(data, status=status.HTTP_200_OK)

