"""FastAPI app that accepts Exotel WSS, transcribes using Whisper/OpenAI, queries OpenAI chat, and returns TTS.

NOTES:
- You must install ffmpeg on the host for audio conversions (pydub uses ffmpeg).
- Adjust the outgoing JSON frame format in send_tts_over_ws() to match Exotel's spec.
"""
import os
import base64
import asyncio
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()
import openai
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

SAVE_DIR = Path("./recordings")
SAVE_DIR.mkdir(exist_ok=True)

app = FastAPI()

# map call_id -> websocket so background tasks can send TTS back
active_connections = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/exotel/callback")
async def exotel_callback(request: Request):
    try:
        form = await request.form()
        data = dict(form)
    except Exception:
        data = await request.json()
    print("[callback] received keys:", list(data.keys())[:10])
    sid = data.get("CallSid") or data.get("call_id") or data.get("CallID") or "unknown"
    (SAVE_DIR / f"{sid}.meta.json").write_text(str(data))
    return PlainTextResponse("OK")

async def run_cmd(cmd):
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return proc.returncode, out, err

def pcm_to_wav(pcm_path, wav_path, sample_rate=8000, channels=1, sample_width=2):
    """Use ffmpeg to convert raw PCM (s16le) to WAV at desired sample rate."""
    # example ffmpeg command:
    # ffmpeg -f s16le -ar 8000 -ac 1 -i input.pcm output.wav
    cmd = f"ffmpeg -y -f s16le -ar {sample_rate} -ac {channels} -i '{pcm_path}' '{wav_path}'"
    res = subprocess.run(cmd, shell=True, capture_output=True)
    if res.returncode != 0:
        print("ffmpeg failed:", res.stderr.decode())
        raise RuntimeError("ffmpeg conversion failed")

def synthesize_tts_pyttsx3(text, out_wav_path):
    """Synthesize TTS using pyttsx3 (offline) and write to out_wav_path."""
    try:
        import pyttsx3
    except Exception as e:
        raise RuntimeError("pyttsx3 not installed")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.save_to_file(text, out_wav_path)
    engine.runAndWait()

async def transcribe_with_openai(wav_path):
    # Uses OpenAI Whisper (via openai.Audio.transcribe) — ensure OPENAI_API_KEY is set
    if not openai_api_key:
        return None
    try:
        with open(wav_path, "rb") as f:
            # The python client historically has openai.Audio.transcribe
            resp = openai.Audio.transcribe("whisper-1", f)
            # resp is likely a dict-like with 'text'
            text = resp.get('text') if isinstance(resp, dict) else getattr(resp, 'text', None)
            return text
    except Exception as e:
        print("OpenAI transcription failed:", e)
        return None

async def chat_with_openai(prompt_text):
    if not openai_api_key:
        return "(no OPENAI_API_KEY configured — reply disabled)"
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content": prompt_text}],
            max_tokens=300,
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        print("OpenAI chat failed:", e)
        return "(error generating reply)"

async def send_tts_over_ws(ws, wav_path):
    """
    Convert wav to raw PCM (s16le) 8kHz mono and stream back as base64 frames over WebSocket.
    **Important**: You must adapt the outgoing JSON frame keys to match Exotel's exact playback schema.
    This implementation sends frames with event 'media' and a direction 'outbound'.
    """
    from pydub import AudioSegment
    seg = AudioSegment.from_file(wav_path)
    seg = seg.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    raw = seg.raw_data
    chunk_size = 3200  # ~200ms of 8kHz 16-bit audio = 1600 samples * 2 = 3200 bytes
    import json
    for i in range(0, len(raw), chunk_size):
        chunk = raw[i:i+chunk_size]
        b64 = base64.b64encode(chunk).decode('ascii')
        frame = {
            "event": "media",
            "direction": "outbound",
            "payload": {"data": b64}
        }
        try:
            await ws.send_text(json.dumps(frame))
            await asyncio.sleep(0.18)
        except Exception as e:
            print("Failed to send TTS chunk:", e)
            break

async def transcribe_and_reply(conn_id, pcm_path, ws=None):
    """Convert PCM->WAV, transcribe, ask OpenAI for reply, synthesize TTS, and send back over ws if available."""
    try:
        wav_tmp = str(Path(pcm_path).with_suffix('.wav'))
        pcm_to_wav(pcm_path, wav_tmp, sample_rate=8000)
    except Exception as e:
        print("Conversion to WAV failed:", e)
        return

    transcript = await transcribe_with_openai(wav_tmp)
    if not transcript:
        transcript = "(no transcript available)"
    print(f"Transcribed call {conn_id}:", transcript)

    reply = await chat_with_openai(transcript)
    print(f"AI reply for {conn_id}:", reply)

    # synthesize to wav
    tts_wav = str(Path(pcm_path).with_name(f"{conn_id}_tts.wav"))
    try:
        synthesize_tts_pyttsx3(reply, tts_wav)
    except Exception as e:
        print("TTS failed:", e)
        return

    # send back over ws
    if ws is not None:
        await send_tts_over_ws(ws, tts_wav)
    else:
        print("No active websocket to send TTS to (call ended or ws missing)")

@app.websocket('/exotel-ws')
async def exotel_ws(websocket: WebSocket):
    await websocket.accept()
    print('WS connected')
    conn_id = None
    audio_file = None
    active = True
    try:
        while True:
            msg = await websocket.receive_text()
            import json
            try:
                obj = json.loads(msg)
            except Exception as e:
                print('Non-json ws message', e)
                continue
            event = obj.get('event') or obj.get('type')
            if event in ('start','call.start'):
                conn_id = obj.get('call_id') or obj.get('callSid') or str(int(asyncio.get_event_loop().time()*1000))
                audio_file = SAVE_DIR / f"{conn_id}.pcm"
                active_connections[conn_id] = websocket
                print('Call started', conn_id)
            elif event in ('media','call.media'):
                payload = obj.get('payload') or {}
                b64 = payload.get('data') or obj.get('data')
                if b64:
                    raw = base64.b64decode(b64)
                    with open(audio_file,'ab') as f:
                        f.write(raw)
            elif event in ('stop','end','call.end'):
                print('Call ended', conn_id)
                # kick off background transcription+reply
                ws_obj = active_connections.pop(conn_id, None)
                # run background task
                asyncio.create_task(transcribe_and_reply(conn_id, str(audio_file), ws=ws_obj))
                # optionally close ws
                try:
                    await websocket.close()
                except:
                    pass
                break
            else:
                print('Other event', event)
    except WebSocketDisconnect:
        print('WS disconnect')
    except Exception as e:
        print('WS error', e)
    finally:
        active_connections.pop(conn_id, None)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
