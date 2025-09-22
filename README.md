# Exotel Voicebot — Repo (with Whisper + OpenAI + TTS demo)

This repo demonstrates a minimal Exotel voicebot:
- accepts Exotel WSS streams, saves incoming PCM audio,
- provides a transcription step using OpenAI Whisper (if `OPENAI_API_KEY` set) or local `whisper` if installed,
- sends the transcribed text to OpenAI Chat (gpt-3.5-turbo) for a reply,
- synthesizes TTS with `pyttsx3` and sends the audio back over the same WebSocket connection (demo format — adapt to Exotel schema).

Files:
- app.py — FastAPI app (WSS + HTTP callback)
- call_test.py — trigger Exotel Calls/connect (test)
- requirements.txt — Python deps
- .env.example — env variables to fill
- Dockerfile — container
- nginx-exotel.conf — example nginx
- service-exotel-voicebot.service — example systemd unit

IMPORTANT:
- This demo assumes Exotel will connect to `wss://<HOST>/exotel-ws` and that Exotel's JSON schema
  for playback accepts a JSON frame with base64 audio. You **must** adapt playback frame keys to Exotel's
  exact spec (see Exotel docs / sample repo).
- For TTS playback we use `pyttsx3` (offline). This requires `ffmpeg` installed to resample/convert.
- For transcription we attempt to use OpenAI's Whisper via `openai` if `OPENAI_API_KEY` is set. Otherwise the code
  falls back to a `whisper` package call if installed locally (optional).

Deployment steps are included in the earlier canvas doc. After extracting the zip, `cp .env.example .env` and fill values.
