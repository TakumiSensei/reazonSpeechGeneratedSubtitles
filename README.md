# ReazonSpeech 音声文字起こしツール

## 概要
ReazonSpeechのモデルを利用して、フォルダ内の音声ファイルを自動で文字起こし（SRT形式）するツールです。  
用途や環境に合わせて、**NeMoモデル** と **K2 V2モデル** の2種類を利用可能です。

- **共通機能**:
  - `inputs/` フォルダ内の音声を自動検出して処理します。
  - `librosa` による無音検出で音声を分割処理し、長い音声でも安定して動作します。
  - `outputs/` フォルダに字幕（SRT）を出力します。
  - GPU (CUDA) が利用可能な場合は自動的に利用し、なければCPUで実行します。

---

## 🔧 環境構築

1. **前提**: Python 3.10 以降、Git、CUDA対応GPU（任意）、FFmpeg/SoundFileなどのオーディオ環境。
2. **仮想環境の作成**:
   ```powershell
   cd C:\workspace\ReazonSpeech
   python -m venv venv
   .\venv\Scripts\activate
   python -m pip install --upgrade pip
   ```
3. **ライブラリのインストール**:
   - リポジトリに同梱されているパッケージ等をインストールしてください。
   - `reazonspeech` ライブラリが必要です。

---

## 🚀 利用方法

### 1. NeMoモデルを使用する場合 (`transcribe_nemo.py`)

ReazonSpeechのNeMoモデルを使用します。

- **特徴**: 高精度な認識が可能。セグメント情報の取得がネイティブに対応。
- **実行方法**:
  ```powershell
  # 仮想環境内で実行
  python transcribe_nemo.py
  ```

### 2. K2 V2モデルを使用する場合 (`transcribe_k2v2.py`)

最新のReazonSpeech K2 V2モデルを使用します。

- **特徴**: Next-gen Kaldi (Sherpa-ONNX) ベースで軽量・高速。
- **実行方法**:
  - 付属のバッチファイル **`run_k2v2.bat`** を実行するだけ（推奨）。
  - または仮想環境内で:
    ```powershell
    python transcribe_k2v2.py
    ```
- **注意点**:
  - K2モデルはネイティブなセグメント情報を持たないため、無音区間で分割したチャンクを1つのセグメントとしてSRT化します。

---

## ⚙️ 設定 (Config)

`config.json` で動作をカスタマイズできます（両モデル共通）。

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `input_dir` | `inputs` | 音声ファイルの読み込み元ディレクトリ |
| `output_dir` | `outputs` | SRTファイルの出力先ディレクトリ |
| `output_mode` | `segment` | `segment`: 発話区間ごと（K2の場合はチャンクごと）<br>`subword`: サブワード単位 |
| `timestamp_format` | `%Y%m%d_%H%M%S` | 出力ファイル名の日時フォーマット |
| `device_preference` | `auto` | `auto`/`cuda`/`cpu` 推論デバイスの指定 |
| `extend_segment_end` | `false` | SRT出力時にセグメントの終了時刻を延長するかどうか。 |
| `extend_segment_end_seconds` | `0.5` | 延長する場合に終了時刻を後ろへどれだけ（秒）ずらすか。次のセグメント開始を超えません。 |
| `supported_extensions` | `.wav`, `.mp3`等 | 対象とする拡張子リスト |

---

## フォルダ構成

- `inputs/`: ここに文字起こししたい音声ファイルを入れてください。
- `outputs/`: ここにSRTファイルが生成されます。
- `venv/`: Python仮想環境（初回作成時）。
