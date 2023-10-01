from faster_whisper import WhisperModel

model_size = "medium.en"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("recorded_audio.wav", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))