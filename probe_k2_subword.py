
import reazonspeech.k2.asr.interface as k2_interface
import inspect

try:
    print(f"Subword fields: {inspect.signature(k2_interface.Subword)}")
except Exception as e:
    print(f"Could not get signature of Subword: {e}")
    # inspect dataclass fields
    if hasattr(k2_interface.Subword, '__dataclass_fields__'):
        print(f"Subword dataclass fields: {k2_interface.Subword.__dataclass_fields__.keys()}")
