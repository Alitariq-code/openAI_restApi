import asyncio
import whisper


def add_timing_to_merged(merged, timing_data):
    # Create a dictionary to store timings for each word to handle duplicate words
    word_timings = {}

    # Iterate through timing data and populate the word_timings dictionary
    for start, end, word in timing_data:
        if word not in word_timings:
            word_timings[word] = {'Start': start, 'End': end}

    # Iterate through merged words and assign timing
    for word_info in merged:
        word = word_info['Word']
        word_id = word_info['ID']  # Get the ID of the word
        if word_info['Class'] == 'Deletion':
            word_info['Timing'] = {}
        else:
            if word in word_timings:
                # Assign timing based on the ID of the word
                if word_id in range(len(timing_data)):
                    start_time, end_time, _ = timing_data[word_id]
                    word_info['Timing'] = {'Start': start_time, 'End': end_time}
                else:
                    word_info['Timing'] = {}
            else:
                word_info['Timing'] = {}

    return merged



word_timings = []
# Load the Whisper model
model = whisper.load_model("base")

async def transcribe_audio(file_path):
    

    print("Transcribing...")
    # Transcribe the audio file with detailed token-level information
    result = model.transcribe(file_path, verbose=True)
    print("Transcription Completed.",result['text'])

    # Iterate over each segment in the transcription
    for segment in result['segments']:
        # The start time of the segment
        segment_start = round(segment['start'], 2)  # Round to 2 decimal places

        # Split segment text into words
        words = segment['text'].split()

        # Calculate the average time for each word
        word_duration = round((segment['end'] - segment['start']) / len(words), 2)  # Round to 2 decimal places

        # Iterate over each word in the segment
        for word in words:
            end_time = round(segment_start + word_duration, 2)  # Round to 2 decimal places
            word_timings.append((segment_start, end_time, word))
            segment_start = end_time
    return result['text'],word_timings
 # Update start time for the next word

    # print(word_timings)

    # # Print word timings
    # for timing_data in word_timings:
    #     print(f"({timing_data[0]:.3f}, {timing_data[1]:.3f}, '{timing_data[2]}'),")
def analyze_speech(data):
    pauses = []
    hesitations = []
    self_corrections = []

    for i, (start_time, end_time, word) in enumerate(data):
        # Check for pauses and hesitations between the current word and the next word
        if i < len(data) - 1:
            next_word_start = float(data[i + 1][0])  # Convert start_time to float
            duration = next_word_start - float(end_time)
            if 1 <= duration <= 3:
                pauses.append((word, duration))
            elif duration > 3:
                hesitations.append((word, duration))
        
        # Check for self-corrections
        if 'correction' in word.lower():  # Assuming corrections are tagged in the transcript
            correction_time = float(start_time)  # Convert start_time to float
            if i > 0:
                prev_word_end = float(data[i - 1][1])  # Convert end_time to float
                if correction_time - prev_word_end <= 3:
                    self_corrections.append((word, correction_time - prev_word_end))

    return pauses, hesitations, self_corrections

# if __name__ == "__main__":
#     # Run the transcription and print the word timings
#     asyncio.run(transcribe_audio("/media/alicode/New_SSD/Speech_Project/Backend/openAI_restApi/temp_original.wav"))
#     # print(word_timings)
#     merged=add_timing_to_merged(merged,word_timings)
#     print(word_timings)
#     pauses, hesitations, self_corrections=analyze_speech(word_timings)
#     print('pauses',pauses,'hesitations', hesitations,'self_corrections', self_corrections)
   
#     # for word_info in merged:
#     #     print(word_info)
    