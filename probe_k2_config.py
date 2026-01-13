
import reazonspeech.k2.asr as k2_asr
import inspect

try:
    print(f"TranscribeConfig fields: {inspect.signature(k2_asr.TranscribeConfig)}")
except Exception as e:
    print(f"Could not get signature: {e}")
    # Try instantiation to see defaults
    try:
        cfg = k2_asr.TranscribeConfig()
        print(f"TranscribeConfig defaults: {cfg}")
    except Exception as e:
        print(f"Could not instantiate: {e}")

try:
    from reazonspeech.k2.asr.interface import TranscribeResult
    print(f"TranscribeResult fields: {TranscribeResult.__annotations__}")
except Exception as e:
    print(f"Could not inspect TranscribeResult: {e}")
