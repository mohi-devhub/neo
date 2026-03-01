class AudioIngestor:
    def _init_(self, model_size="base"):
        import whisper
        self.model = whisper.load_model(model_size)

    def _transcribe_(self, audio_path):
        result = self.transcribe(audio_path)
        print(result["text"])