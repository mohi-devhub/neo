import os
import subprocess
import tempfile
from typing import Optional


class VideoIngestor:

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            output_path = tmp.name
            tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path,
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed: {result.stderr.decode(errors='replace')}"
            )

        return output_path

    def process_video_with_audio(
        self,
        video_path: str,
        filename: str,
        audio_ingestor,
        language: Optional[str] = None,
    ):
        """
        Returns:
            transcript          (str)
            transcript_segments (list of Whisper segment dicts with start_time/end_time)
        """
        transcript = ""
        transcript_segments = []

        if audio_ingestor is not None and audio_ingestor.model is not None:
            audio_path = None
            try:
                audio_path = self.extract_audio(video_path)
                result = audio_ingestor.transcribe(audio_path, language=language)
                transcript = str(result.get("text", ""))
                transcript_segments = result.get("segments", [])
            except Exception as e:
                print(f"[VideoIngestor] audio transcription failed (non-fatal): {e}")
            finally:
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.unlink(audio_path)
                    except OSError:
                        pass

        return transcript, transcript_segments
