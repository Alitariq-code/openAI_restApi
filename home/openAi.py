import timeit
import whisper
import asyncio
model = None
async def load_model():
    global model
    # Load the model asynchronously
    model = await asyncio.to_thread(whisper.load_model, "tiny")

async def transcribe_audio(file_path):
    if model is None:
        await load_model()
    # Transcribe the audio file
    result = await asyncio.to_thread(model.transcribe, file_path)
    return result["text"]

# # Asynchronously transcribe audio
# async def main():
#     transcription = await transcribe_audio("AVI 7.wav")
#     print(transcription)

# start = timeit.default_timer()
# # Run the main function
# asyncio.run(main())
# stop = timeit.default_timer()
# print('Time: ', stop - start)  