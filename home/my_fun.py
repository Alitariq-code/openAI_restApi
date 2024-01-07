from pydub import AudioSegment
from pydub.silence import split_on_silence
from difflib import SequenceMatcher
import pandas as pd
import whisper
import re

from fuzzywuzzy import fuzz

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
    def preprocess_text(text):
        return re.sub(r'[\r\n]+', ' ', re.sub(r'[^\w\s]', ' ', text)).lower()
    def align_texts(original, spoken):
        sequence_matcher = SequenceMatcher(None, original, spoken)
        operations = []
        for opcode in sequence_matcher.get_opcodes():
            tag, i1, i2, j1, j2 = opcode
            if tag == 'delete':
                operations.append(['Deletion', ' '.join(original[i1:i2]), i1, i2-1, '', '', '', '', '', 0])
            elif tag == 'insert':
                before_context = ' '.join(spoken[max(0, j1-1):j1])
                after_context = ' '.join(spoken[j2:min(len(spoken), j2+1)])
                operations.append(['Insertion', '', '', '', ' '.join(spoken[j1:j2]), j1, j2-1, before_context, after_context, 0])
            elif tag == 'replace':
                original_segment = ' '.join(original[i1:i2])
                spoken_segment = ' '.join(spoken[j1:j2])
                similarity = fuzz.ratio(original_segment, spoken_segment)
                operations.append(['Substitution', original_segment, i1, i2-1, spoken_segment, j1, j2-1, '', '', similarity])
        return operations
    def process_substitutions(df):
        new_rows = []
        for index, row in df.iterrows():
            if row['Operation'] == 'Substitution' and row['Similarity'] < 40:
                # Splitting the row into two parts: Insertion and Deletion
                insertion_row = row.copy()
                insertion_row['Operation'] = 'Insertion'
                deletion_row = row.copy()
                deletion_row['Operation'] = 'Deletion'
                new_rows.extend([insertion_row, deletion_row])
            else:
                new_rows.append(row)
        return pd.DataFrame(new_rows)
    # Preprocessing texts
    preprocessed_original_text = preprocess_text(original_text).split()
    preprocessed_spoken_text = preprocess_text(spoken_text).split()
    # Aligning and getting differences
    differences = align_texts(preprocessed_original_text, preprocessed_spoken_text)
    # Creating DataFrame
    df = pd.DataFrame(differences, columns=['Operation', 'Original Segment', 'Original Start', 'Original End', 'Spoken Segment', 'Spoken Start', 'Spoken End', 'Before Context', 'After Context', 'Similarity'])
    # Process substitutions
    processed_df = process_substitutions(df)
    # Function to select columns based on the operation
    def select_columns(row):
        if row['Operation'] in ['Insertion', 'Substitution']:
            return pd.Series([row['Operation'], row['Spoken Segment'], row['Spoken Start'], row['Spoken End']])
        elif row['Operation'] == 'Deletion':
            return pd.Series([row['Operation'], row['Original Segment'], row['Original Start'], row['Original End']])
    # Define a function to determine the class based on the color
    def determine_class(color):
        if color == 'Red':
            return 'Deletion'
        elif color == 'Yellow':
            return 'Substitution'
        elif color == 'Pink':
            return 'Insertion'
        elif color == 'Green':
            return 'Correct'
        else:
            return 'Unknown'

    # Function to generate ID and word pairs
    def generate_id_word_pairs(row):
        start = row['Start/Spoken Start']
        end = row['End/Spoken End']
        segment = row['Segment/Original Segment']
        # Split the segment into words
        words = segment.split()
        # Generate ID and word pairs
        id_word_pairs = [(start + i, word) for i, word in enumerate(words)]
        return id_word_pairs
    try:
        # Applying the function to each row and reformating DataFrame
        reformatted_df = processed_df.apply(select_columns, axis=1)
        reformatted_df.columns = ['Operation', 'Segment/Original Segment', 'Start/Spoken Start', 'End/Spoken End']
        reformatted_df=reformatted_df.reset_index(drop=True)
        insert=reformatted_df[reformatted_df['Operation']=='Insertion']
        delete=reformatted_df[reformatted_df['Operation']=='Deletion']
        subt=reformatted_df[reformatted_df['Operation']=='Substitution']
        # Apply the function to each row and flatten the list
        try:
            id_word_pairs_list = insert.apply(generate_id_word_pairs, axis=1).explode()
            # Create a new dataframe with ID and word columns
            insert_df = pd.DataFrame(id_word_pairs_list.tolist(), columns=['ID', 'word'])
        except:
            insert_df = pd.DataFrame(columns=['ID', 'word'])
        try:
            # Apply the function to each row and flatten the list
            id_word_pairs_list = delete.apply(generate_id_word_pairs, axis=1).explode()
            # Create a new dataframe with ID and word columns
            delete_df = pd.DataFrame(id_word_pairs_list.tolist(), columns=['ID', 'word'])
        except:
            delete_df= pd.DataFrame(columns=['ID', 'word'])
        try:
            id_word_pairs_list = subt.apply(generate_id_word_pairs, axis=1).explode()
            # Create a new dataframe with ID and word columns
            subt_df = pd.DataFrame(id_word_pairs_list.tolist(), columns=['ID', 'word'])
        except:
            subt_df= pd.DataFrame(columns=['ID', 'word'])
        # Iterate over each row in the Substitution DataFrame
        substituted_words = []
        delete_words = []
        insert_words = []
        for index, row in subt_df.iterrows():
            substituted_word = {
            'ID': row['ID'],
            'Word': row['word'],
        }
            substituted_words.append(substituted_word)
        for index, row in delete_df.iterrows():
            delete_word = {
            'ID': row['ID'],
            'Word': row['word'],
        }
            delete_words.append(delete_word)
        for index, row in insert_df.iterrows():
            insert_word = {
            'ID': row['ID'],
            'Word': row['word'],
        }
            insert_words.append(insert_word)
    # ...
        print("subWOrds",substituted_words)
        print("deleted",delete_words)
        print("insert",insert_words)
    except:
        delete_df= pd.DataFrame(columns=['ID', 'word'])
        subt_df= pd.DataFrame(columns=['ID', 'word'])
        insert_df= pd.DataFrame(columns=['ID', 'word'])
        substituted_words = []
        delete_words = []
        insert_words = []
        for index, row in subt_df.iterrows():
            substituted_word = {
            'ID': row['ID'],
            'Word': row['word'],
        }
            substituted_words.append(substituted_word)
        for index, row in delete_df.iterrows():
            delete_word = {
            'ID': row['ID'],
            'Word': row['word'],
        }
            delete_words.append(delete_word)
        for index, row in insert_df.iterrows():
            insert_word = {
            'ID': row['ID'],
            'Word': row['word'],
        }
            insert_words.append(insert_word)
    # ...
        print("subWOrds",substituted_words)
        print("deleted",delete_words)
        print("insert",insert_words)
    spoken_df = pd.DataFrame({'ID': range(len(preprocessed_spoken_text)), 'word': preprocessed_spoken_text})
    spoken_df['Color']='Green'
    merged_df = pd.merge(spoken_df, subt_df, on='ID', how='left', suffixes=('_spoken', '_subt'))
    merged_df['Color'] = merged_df['Color'].where(merged_df['word_subt'].isnull(), 'Yellow')
    merged_df=merged_df[['ID','word_spoken','Color']]
    merged_df = pd.merge(merged_df, insert_df, on='ID', how='left', suffixes=('', '_insr'))
    merged_df['Color'] = merged_df['Color'].where(merged_df['word'].isnull(), 'Pink')
    merged_df=merged_df[['ID','word_spoken','Color']]
    delete_df['Color']='Red'
    delete_df.columns=merged_df.columns
    merged_df=pd.concat([merged_df,delete_df])
    merged_df=merged_df.sort_values("ID")
    merged_df=merged_df.reset_index().reset_index()
    merged_df=merged_df[['level_0','word_spoken','Color']]
    merged_df.columns=delete_df.columns
    # Create a new column 'Class' based on the 'Color' column
    merged_df['Class'] = merged_df['Color'].apply(determine_class)
    merged = [] 
    for index, row in merged_df.iterrows():
            insert_word = {
            'ID': row['ID'],
            'Word': row['word_spoken'],
            'Color': row['Color'],
            'Class': row['Class'],
        }
            merged.append(insert_word)
    return substituted_words,delete_words,insert_words,merged
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

def calculate_error_metrics(original_text, transcribed_text, delete, insert, sub, manual_text):
    words_original = original_text.split()
    words_transcribed = transcribed_text.split()

    if manual_text:
        words_manual_text = manual_text.split()
        wt_maual = len(words_manual_text)
    else:
        words_manual_text = []
        wt_maual = 0

    wr = len(words_transcribed)
    wt_original = len(words_original)
    wc = wt_original - (delete + insert + sub)

    acc = wc * 100 / wt_original
    wc = max(0, wc)
    acc = max(0, acc)
    oriVsTran = 100 * min(wt_original / wr, wr / wt_original)
    manualVsTrans = 100 * min(wt_maual / wr, wr / wt_maual) if wt_maual > 0 else 0
    manualVsorginal = 100 * min(wt_maual / wt_original, wt_original / wt_maual) if wt_maual > 0 else 0

    error_metrics = {
        'WR': wr,
        'WC': wc,
        'Words Correct per Minute': calculate_words_per_minute(wc, transcribed_text),
        'Acc': acc,
        'oriVsTran': oriVsTran,
        'manualVsTrans': manualVsTrans,
        'manualVsorginal': manualVsorginal
    }

    return error_metrics
def calculate_word_count_ratio(transcribed_text, original_text, max_ratio=100):
    word_count_org = count_words(original_text)
    word_count_transcribed = count_words(transcribed_text)

    ratio = min(100*word_count_org / word_count_transcribed , word_count_transcribed* 100/word_count_org)
    return ratio

# def transcribe_audio(file_path):
#         try:
#             # Load the Whisper model
#             model = whisper.load_model("base")
#             # Transcribe the audio file
#             result = model.transcribe(file_path)

#             # Check if the result is a string
#             if isinstance(result, str):
#                 return result
#             elif isinstance(result, dict):
#                 # Return the transcription if available, otherwise an empty string
#                 return result.get("text", "")
#             else:
#                 raise ValueError("Unexpected result type from transcribe method")

#         except Exception as e:
#             print(f"Error in transcribe_audio: {str(e)}")
#             return ""


# Example usage:
# transcribed_text = transcribe_audio("/home/alicode/Desktop/stock/AVI 4.wav")
# duration = get_wav_duration('temp.wav')
# analysis_result = analyze_audio('temp.wav')
# deleted_words, inserted_words, substituted_words, repeated_words = compare_lines(org, transcribed_text)
# dup = count_duplicate_lines(transcribed_text)
# skip = count_skipped_lines(transcribed_text)
# word_count = count_words(transcribed_text)
# ...
