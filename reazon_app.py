import sys
from app_utils import load_app_config, get_base_path

def main():
    try:
        # Load config from the executable's directory
        base_path = get_base_path()
        config = load_app_config(base_path)
        
        print(f"ReazonSpeech Standalone App")
        print(f"Base Directory: {base_path}")
        print(f"Model Type: {config.model_type}")
        print(f"Remove Period: {config.remove_period}")
        
        if config.model_type == "nemo":
            print("Importing NeMo module...")
            import transcribe_nemo
            transcribe_nemo.run_batch(config, base_path)
        else:
            print("Importing K2V2 module...")
            import transcribe_k2v2
            transcribe_k2v2.run_batch(config, base_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nProcess finished. Press Enter to exit.")
    input()

if __name__ == "__main__":
    main()
