# 環境構築ガイド (Windows)

TakumiSensei/reazonSpeechGeneratedSubtitles プロジェクトをローカルで動かすための環境構築手順です。

## 1. 前提条件 (Prerequisites)

以下のツールがインストールされていることを確認してください。

*   **Python 3.10 以上**: [Python公式サイト](https://www.python.org/downloads/)からインストール (インストール時に "Add Python to PATH" にチェックを入れてください)
*   **Git**: [Git公式サイト](https://git-scm.com/download/win)からインストール
*   **FFmpeg**: 音声処理に必要です。
    *   インストール方法 (Wingetを使用する場合):
        ```powershell
        winget install Gyan.FFmpeg
        ```
    *   または公式サイトからダウンロードし、binフォルダへのパスを環境変数Pathに通してください。
*   **CUDA Toolkit** (NVIDIA GPUを使用する場合): GPUドライバと適合するバージョンをインストールしてください (通常はPyTorchが対応するバージョン、例: 11.8 や 12.1)。

## 2. リポジトリのクローン

PowerShellまたはターミナルを開き、任意のディレクトリで以下を実行します。

```powershell
git clone --recursive https://github.com/TakumiSensei/reazonSpeechGeneratedSubtitles.git
cd reazonSpeechGeneratedSubtitles
```
※ `--recursive` はサブモジュール (`ReazonSpeech` フォルダなど) を含めて取得するために推奨されますが、このリポジトリの構造によっては必須ではない場合もあります。

## 3. 仮想環境の作成と有効化

プロジェクト専用のPython環境を作成します。

```powershell
# 仮想環境 (venv) の作成
python -m venv venv

# 仮想環境の有効化 (Windows)
.\venv\Scripts\activate
```

有効化されるとプロンプトの先頭に `(venv)` と表示されます。

## 4. ライブラリのインストール

### 4.1. 基本ツールのアップグレード

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 4.2. PyTorchのインストール
**重要**: GPUを使用する場合は、CUDA対応版をインストールする必要があります。
(CPUのみの場合は `pip install torch torchaudio` で十分ですが、GPU推奨です)

CUDA 11.8 の場合 (例):
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```
※ お使いのCUDAバージョンに合わせて `--index-url` を変更してください ([PyTorch公式サイト参照](https://pytorch.org/get-started/locally/))。

### 4.3. ReazonSpeech と依存ライブラリのインストール

```powershell
# requirements.txt を使う場合 (推奨)
pip install -r requirements.txt

# または個別にインストールする場合
pip install reazonspeech numpy librosa soundfile

# NeMoモデルを使用する場合 (transcribe_nemo.py) は以下も推奨
pip install "nemo_toolkit[asr]"

# K2 V2モデル (transcribe_k2v2.py) は reazonspeech に含まれる sherpa-onnx 等を利用します
```

※ もし `ReazonSpeech` フォルダ内のローカルパッケージを使用する必要がある場合は、以下のようにインストールします (通常はPyPI版で動作します)。
```powershell
pip install ./ReazonSpeech/pkg/nemo-asr
pip install ./ReazonSpeech/pkg/k2-asr
```

## 5. 動作確認

### 設定ファイルの確認
`config.json` を開き、必要に応じて設定を変更できます (デフォルトのままでも動作します)。

### 実行
音声ファイル (`.mp3`, `.wav` 等) を `inputs` フォルダに配置し、以下のバッチファイルをダブルクリックまたはターミナルから実行します。

*   **NeMoモデル (高精度)**: `run_nemo.bat`
*   **K2 V2モデル (高速)**: `run_k2v2.bat`

初回実行時はモデルのダウンロードが行われるため時間がかかります。完了すると `outputs` フォルダに字幕ファイル (SRT) が生成されます。
