from pydub import AudioSegment
from pydub.silence import split_on_silence
from difflib import SequenceMatcher
import pandas as pd
import whisper

def get_wav_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_in_seconds = len(audio) / 1000.0
    return duration_in_seconds

def analyze_audio(file_path):
    audio = AudioSegment.from_file(file_path, format="wav")

    short_pause_threshold = 3000  # 3 seconds
    long_pause_threshold = 3000   # 3 seconds
    word_repetitions, short_pauses, long_pauses = 0, 0, 0

    segments = split_on_silence(audio, silence_thresh=-40)

    for i in range(len(segments)):
        segment_duration = len(segments[i])

        if short_pause_threshold <= segment_duration <= long_pause_threshold:
            short_pauses += 1
        elif segment_duration > long_pause_threshold:
            long_pauses += 1

        window_start = max(0, i - 1)
        window_end = min(len(segments), i + 1)
        window = segments[window_start:window_end]

        if len(window) > 1 and compare_segments(segments[i], sum(window)):
            word_repetitions += 1

    return {
        "word_repetitions": word_repetitions,
        "short_pauses": short_pauses,
        "long_pauses": long_pauses
    }

def remove_newlines(text):
    cleaned_text = text.replace('\n', '')
    return cleaned_text

def remove_punctuation(text):
    cleaned_text = text.replace(',', '').replace('.', '').replace('“', '').replace('”', '')
    return cleaned_text

def track_deleted_words(original_lines, spoken_lines):
    original_lines = remove_newlines(original_lines)
    original_lines = remove_punctuation(original_lines)
    spoken_lines = remove_newlines(spoken_lines)
    spoken_lines = remove_punctuation(spoken_lines)

    original_words = original_lines.split()
    spoken_words = spoken_lines.split()

    deleted_words = []
    deleted_positions = []

    for index, word in enumerate(original_words):
        if word not in spoken_words:
            deleted_words.append(word)
            deleted_positions.append(index)

    return deleted_words, deleted_positions

def track_inserted_words(original_lines, spoken_lines):
    original_lines = remove_newlines(original_lines)
    original_lines = remove_punctuation(original_lines)
    spoken_lines = remove_newlines(spoken_lines)
    spoken_lines = remove_punctuation(spoken_lines)

    original_words = original_lines.split()
    spoken_words = spoken_lines.split()

    inserted_words = []
    inserted_positions = []

    for index, word in enumerate(spoken_words):
        if word not in original_words:
            inserted_words.append(word)
            inserted_positions.append(index)

    return inserted_words, inserted_positions

def find_most_similar(word, word_list):
    similarities = [(w, SequenceMatcher(None, word, w).ratio()) for w in word_list]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0]

def compare_lines(original_text, spoken_text):
    deleted_words, deleted_positions = track_deleted_words(original_text, spoken_text)
    inserted_words, inserted_positions = track_inserted_words(original_text, spoken_text)
    
    df_insert = pd.DataFrame(inserted_words, inserted_positions).reset_index()
    df_insert.columns = ['ID', 'Word']

    df_delete = pd.DataFrame(deleted_words, deleted_positions).reset_index()
    df_delete.columns = ['ID', 'Word']

    result_data = {'Delete Word': [], 'Insert Word': [], 'Similarity Score': []}

    for index_delete, row_delete in df_delete.iterrows():
        similar_word, similarity_score = find_most_similar(row_delete['Word'], df_insert['Word'])
        result_data['Delete Word'].append(row_delete['Word'])
        result_data['Insert Word'].append(similar_word)
        result_data['Similarity Score'].append(similarity_score)

    result_df = pd.DataFrame(result_data)

    sub_df = result_df[result_df['Similarity Score'] > 0.55]
    df_delete = df_delete[~df_delete['Word'].isin(sub_df['Delete Word'])]
    df_substitute = df_insert[df_insert['Word'].isin(sub_df['Insert Word'])]
    df_insert = df_insert[~df_insert['Word'].isin(sub_df['Insert Word'])]

    print("deleted", df_delete)
    return df_delete, df_substitute, df_insert

def count_duplicate_lines(text_data):
    seen_lines = set()
    duplicate_count = 0

    lines = text_data.split('\n')

    for line in lines:
        line = line.strip()
        if line:
            if line in seen_lines:
                duplicate_count += 1
            else:
                seen_lines.add(line)

    return duplicate_count

def count_skipped_lines(text_data):
    lines = text_data.split('\n')
    skipped_count = 0

    for i in range(1, len(lines) - 1):
        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()

        if not next_line:
            skipped_count += 1

    return skipped_count

def count_words(text):
    words = text.split()
    return len(words)

def remove_duplicates(word_list):
    unique_words = []
    seen_words = set()

    for word in word_list:
        if word not in seen_words:
            unique_words.append(word)
            seen_words.add(word)

    return unique_words

def remove_newlines(text):
    return text.replace('\n', '')

def compare_segments(segment1, segment2):
    return segment1.set_frame_rate(44100).set_channels(1).rms > segment2.set_frame_rate(44100).set_channels(1).rms * 0.8

def find_repeated_words(text):
    words = text.split()
    repeated_words = []

    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            repeated_words.append(words[i])

    return repeated_words

def calculate_words_per_minute(words_correct, transcribed_text):
    audio_duration = get_wav_duration('temp.wav')
    words_per_minute = (words_correct / audio_duration) * 60
    return words_per_minute

def calculate_error_metrics(original_text, transcribed_text):
    words_original = original_text.split()
    words_transcribed = transcribed_text.split()

    wc = len(set(words_original) & set(words_transcribed))
    wr = len(words_transcribed)

    error_metrics = {
        'WR': wr,
        'WC': wc,
        'Words Correct per Minute': calculate_words_per_minute(wc, transcribed_text),
    }

    return error_metrics

def calculate_word_count_ratio(transcribed_text, original_text, max_ratio=100):
    word_count_org = count_words(original_text)
    word_count_transcribed = count_words(transcribed_text)

    ratio = min(word_count_org / word_count_transcribed * 100, max_ratio)
    return ratio


def transcribe_audio(file_path):
        try:
            # Load the Whisper model
            model = whisper.load_model("base")
            # Transcribe the audio file
            result = model.transcribe(file_path)

            # Check if the result is a string
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                # Return the transcription if available, otherwise an empty string
                return result.get("text", "")
            else:
                raise ValueError("Unexpected result type from transcribe method")

        except Exception as e:
            print(f"Error in transcribe_audio: {str(e)}")
            return ""


# Example usage:
# transcribed_text = transcribe_audio("/home/alicode/Desktop/stock/AVI 4.wav")
# duration = get_wav_duration('temp.wav')
# analysis_result = analyze_audio('temp.wav')
# deleted_words, inserted_words, substituted_words, repeated_words = compare_lines(org, transcribed_text)
# dup = count_duplicate_lines(transcribed_text)
# skip = count_skipped_lines(transcribed_text)
# word_count = count_words(transcribed_text)
# ...
