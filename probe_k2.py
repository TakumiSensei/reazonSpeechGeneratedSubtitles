
import sys
try:
    import reazonspeech.k2.asr as k2_asr
    print("reazonspeech.k2.asr imported successfully")
except ImportError:
    print("reazonspeech.k2.asr import failed")
    sys.exit(1)

print(f"k2_asr attributes: {dir(k2_asr)}")

if hasattr(k2_asr, 'load_model'):
    import inspect
    print(f"load_model signature: {inspect.signature(k2_asr.load_model)}")

if hasattr(k2_asr, 'transcribe'):
    import inspect
    print(f"transcribe signature: {inspect.signature(k2_asr.transcribe)}")
