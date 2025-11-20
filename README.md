# ReazonSpeech 音声文字起こしツール

## 概要
- `transcribe_nemo.py` は ReazonSpeech の NeMo ASR モデルを利用し、`inputs/` 以下の音声ファイルを検出して自動文字起こしを行うスクリプトです。
- 音声は `librosa.effects.split` を使って無音区間でチャンクに分割され、長時間ファイルでも安定して推論できるようになっています。
- 認識結果は字幕形式（SRT）で `outputs/` に書き出され、出力内容はセグメント単位またはサブワード単位を設定で切り替えられます。
- CUDA/GPU が利用できる場合は自動で GPU を選択し、利用不可の場合は CPU へフォールバックします。

## 環境構築
1. 前提: Python 3.10 以降、Git、CUDA 対応 GPU（任意）、FFmpeg/SoundFile が読み込めるオーディオコーデック環境。
2. 仮想環境を作成して有効化します。
   ```powershell
   cd C:\workspace\ReazonSpeech
   python -m venv venv
   .\venv\Scripts\activate
   python -m pip install --upgrade pip
   ```
3. ReazonSpeech の NeMo ASR パッケージをローカルインストールして依存関係（torch, nemo_toolkit, librosa など）を導入します。
   ```powershell
   pip install -e .\ReazonSpeech\pkg\nemo-asr
   ```
4. モデル初回実行時に NeMo モデルがダウンロードされるため、十分なディスク容量とネットワーク帯域を確保してください。

## プログラム説明
- 実行コマンド: `python transcribe_nemo.py`
- スクリプトは起動時に `config.json` を読み込み、入力・出力ディレクトリを自動生成します。`inputs/` 内の `supported_extensions` に合致するファイルのみが処理対象です。
- `select_device` で `device_preference` に従って GPU/CPU を選択し、`reazonspeech.nemo.asr.load_model` で推論モデルを 1 度だけロードします。
- 各音声は `split_audio_on_silence` により無音区間で分割され、`transcribe` にチャンクごと渡して高速化します。分割結果は `merge_transcriptions` で結合され、文字列・セグメント・サブワード単位のタイムスタンプを保持したまま統合されます。
- 出力は `segments_to_srt` もしくは `subwords_to_srt` で SRT 文字列に整形され、`<元ファイル名>_<timestamp>.srt` という形式で `outputs/` に保存されます。処理進捗やチャンク数は標準出力に表示されます。

## コンフィグ内容
`config.json` で以下の項目を調整できます。未記載の項目は既定値が使用され、保存後に次回実行から反映されます。

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `input_dir` | `inputs` | 読み込む音声ファイルを配置する相対ディレクトリ。 |
| `output_dir` | `outputs` | SRT を書き出すディレクトリ。存在しない場合は自動作成。 |
| `output_mode` | `segment` | `segment` はモデルの発話セグメント単位で字幕化、`subword` はサブワードごとの細切れ字幕を生成。 |
| `timestamp_format` | `%Y%m%d_%H%M%S` | 出力ファイル名に付与される日時フォーマット（`datetime.strftime` 準拠）。 |
| `device_preference` | `auto` | `auto`/`cuda`/`cpu` で推論デバイスを指定。`cuda` 選択時に GPU が無い場合は CPU へ自動フォールバック。 |
| `extend_segment_end` | `false` | SRT 出力時にセグメントの終了時刻を延長するかどうか。 |
| `extend_segment_end_seconds` | `0.5` | 延長する場合に終了時刻を後ろへどれだけ（秒）ずらすか。次のセグメント開始を超えません。 |
| `supported_extensions` | `[".wav", ".mp3", ".flac", ".m4a", ".ogg"]` | 処理対象とする拡張子一覧。`.` 付き小文字で比較され、重複は除去されます。 |

`config.json` を調整した後は、`inputs/` に音声を置いて再度 `python transcribe_nemo.py` を実行するだけで設定の違いを反映した字幕ファイルを生成できます。
