# CosyVoice3 原生 LoRA Voice Overlay 訓練與部署方案

本文件說明如何在 `cosyvoice.cpp` 生態中，針對單一角色或特定聲線（約 30–200 句、3–20 分鐘乾淨語音），訓練輕量級原生 **F32 LoRA Voice Overlay**（`cosyvoice3-voice-overlay-v1`，單角色約 `33.6 MB`），並搭配**共用混合精度底座**與 **F16 KV Cache** 進行高品質推論部署。

> **適用範圍**：對應 `CosyVoice3-2512`（Qwen2-0.5B Speech-Token LLM + Flow/DiT + HiFT）與支援 `--voice-adapter`、`--voice-adapter-scale`、`--llm-kv-cache-type` 的 `cosyvoice.cpp` 執行環境。對應腳本位於 [`tools/voice_overlay_training/`](../tools/voice_overlay_training/README.md)。

---

## 1. 架構與設計原則

### 1.1 為什麼只微調 Speech-LLM，而凍結 Flow（DiT）與 HiFT？

CosyVoice3 的合成管線由三個模組組成：

| 模組 | 職責 | 小樣本角色微調策略 |
| :--- | :--- | :--- |
| **Speech-LLM**（24 層 Qwen2-0.5B） | 文字＋指令 → 25 Hz 離散語音 Token，主導**語氣、斷句、停頓節奏、重音與跨語種自然度** | **套用 Rank-16 LoRA**（7 個線性層全開），凍結 `embed_tokens` 與 `speech_embedding` / `llm_decoder` |
| **Flow / DiT** | 語音 Token＋參考音色 Embedding → Mel 頻譜，主導**音色質感與聲學細節** | **凍結**（小樣本動 Flow 容易出現 `grad_norm nan`、機械音或語意漂移；音色由乾淨 `prompt_speech.gguf` 搭配 LLM 語音 Token 分布共同對齊） |
| **HiFT Vocoder** | Mel 頻譜 → 24 kHz 波形 | **凍結**，並在共用底座中保留原生 **F32** 精度確保高頻齒擦音與弱音細節 |

> **邊界說明**：對於大多數清晰人聲與配音角色，只微調 LLM LoRA 搭配高品質參考音已可同時兼顧音色貼合與跨語種自然度。僅在遇到光靠零樣本參考音完全無法接近的極端特殊聲線且擁有足夠語料（數小時以上）時，才需考慮額外微調 Flow。

### 1.2 共用混合精度底座 ＋ 原生 F32 LoRA Overlay

相較為每個角色各存一份全量模型（約 `1.72 GB / 角色`）或全張量覆寫檔（約 `~715 MB / 角色`），本方案採用：

1. **音質優先混合精度共用底座（`CosyVoice3-2512_F16_HiFT-F32.gguf`）**：
   - 使用 `build_mixed_base.py` 從官方 `F32` GGUF（可選配官方 `F16` GGUF 逐位對照）建置：
     - `layers.*`（LLM 24 層）、`decoder.*`（Flow/DiT）、CPU mmap 之 `embed_tokens.*` 維持 **F16**，與標準官方 F16 底座 `layers.*` 逐位元一致，確保所有 Voice Overlay 與動態 LoRA 100% 位元相容。
     - HiFT 聲碼器、F0 預測器、`llm_decoder`、`speech_embedding` 與所有 1D norm/bias 保留原生 **F32**（僅較標準 F16 增加約 `65 MB`）。
   - 若不想重打包，亦可直接沿用標準官方 `F16` 底座（注意：被 LoRA 覆寫的 `layers.*` 必須為 `F16` 或 `F32`，不可為 `Q8_0` 等區塊量化型別）。
2. **輕量原生低秩 F32 Overlay（`cosyvoice3-voice-overlay-v1`）**：
   - 直接由訓練過程的 `lora_A` / `lora_B` 矩陣寫出，不需先合併再做近似 SVD 反推。
   - `r=16`、F32 精度單檔約 **`33.6 MB`**；載入時由原生引擎在 FP32 中與底座權重合併後再做 QKV / GateUp 融合並上傳裝置，推論期**零額外計算開銷**。
   - 支援在推論期透過 `--voice-adapter-scale`（如 `0.55`、`0.70`、`0.85`）微調「角色風格強度 vs 多語種泛化」平衡，無需重訓。
3. **執行期 F16 LLM KV Cache**：
   - 推論時加上 `--llm-kv-cache-type f16`，避免預設非對稱量化（如 `k=q8_0,v=q4_0`）對長句柔聲語氣造成的細微壓縮毛躁。

---

## 2. 環境與前置素材準備

### 2.1 軟硬體需求
- **推論與前處理執行期**：已建置或下載之 `cosyvoice.cpp`（含 `cosyvoice-cli`、`cosyvoice-server`、原生共用函式庫 `cosyvoice.dll` / `libcosyvoice.so` / `libcosyvoice.dylib` 與對應 GGML 1.x 函式庫）。
- **模型檔案**：
  - 官方 `CosyVoice3-2512_F32.gguf`（用於純 FP32 訓練與建置混合精度底座）
  - （選填）官方 `CosyVoice3-2512_F16.gguf`（供 `build_mixed_base.py --f16` 驗證 `layers.*` 逐位一致）
  - `speech_tokenizer_v3.onnx`、`campplus.onnx`
- **Python 訓練環境**：Python 3.10+、支援 CUDA 或 ROCm 的 PyTorch（訓練 0.5B LLM FP32 + LoRA 僅需約 **5–6 GB VRAM**）、`ffmpeg` / `ffprobe`（若輸入含非 24 kHz 單聲道 WAV）。

```bash
python -m pip install -r tools/voice_overlay_training/requirements.txt
```

### 2.2 原始語音素材規格與清洗原則

從多角色實測經驗來看，**資料與參考音品質對最終聽感的影響遠大於微調 LoRA 超參**。訓練前請優先把關以下項目：

| 項目 | 建議標準 |
| :--- | :--- |
| **數量與總時長** | **30–200 句**（約 3–20 分鐘乾淨人聲）；重質不重量，寧缺勿濫 |
| **單句長度** | **1.2–18 秒**（最理想 3–10 秒完整自然語句） |
| **錄音品質** | 乾淨單人聲、無 BGM、無回音、無削波破音 |
| **剔除清單** | 明顯錯字、口吃斷尾、無實義擬聲／純喘氣／笑聲、含舞台指示符號（如 `(動作)`、`♪`、`♥`、連續省略號）之樣本 |
| **支援輸入格式** | 同名 `<stem>.wav` + `<stem>.txt`（可選 `<stem>.inst.txt`），或單一清單檔 `metadata.jsonl` / `transcripts.txt` / `metadata.csv` |

#### 關鍵實戰提醒 A：近距離錄音句間大吸氣（`debreath_audio.py`）
部分近距離配音素材在逗號停頓處帶有響亮的吸氣嘶聲。若原封不動拿去訓練，即使推論時換成乾淨參考音，模型在 `instruct` 或長句生成時仍會在逗號處主動學出大口喘氣。

建議先以 `--dry-run` 偵測，確認比例偏高再輸出至獨立乾淨目錄（**絕不覆寫原始素材**）：

```bash
# 先乾跑統計偵測到的句間吸氣數
python tools/voice_overlay_training/debreath_audio.py --in-dir ./data/raw --dry-run

# 輸出至新目錄並同步複製 .txt / 清單（僅針對鄰近靜音帶的 1.2–4.5 kHz 無基頻氣音做餘弦衰減，保留母音與齒擦音）
python tools/voice_overlay_training/debreath_audio.py \
  --in-dir ./data/raw --out-dir ./data/raw_debreathed --copy-sidecars
```

處理後務必抽聽吸氣最多的 5–8 句，確認所有母音與 `s/sh/ch/ts` 齒擦音完整無截切；亦可在 `train_config.json` 中改用 `audio_preprocess.debreath_enabled: true` 整合執行。

#### 關鍵實戰提醒 B：日語文本與參考音標記一致原則
- 若你的推論前端會將日文漢字正規化為片假名（例如搭配 SudachiPy 讀音展開），**所有日語訓練逐字稿與參考音 `prompt_speech.gguf` 的 `--prompt-text` 必須統一為相同片假名格式**。
- 常見踩坑：訓練集已換成片假名，但推論時誤用舊的漢字版 `prompt_speech.gguf`，會導致 `zero-shot` 日文出現吃字、拆詞或誤讀。每次更新參考音或逐字稿後，請務必重新執行 Stage 1 並確認 `recommended_reference.gguf` 已更新。

---

## 3. 完整五階段操作流程

先複製範例設定並填入對應路徑：

```bash
cp tools/voice_overlay_training/train_config.example.json train_config.json
```

### Stage 0：建置混合精度共用底座（一次性）

```bash
python tools/voice_overlay_training/build_mixed_base.py \
  --f32 ./models/CosyVoice3-2512_F32.gguf \
  --f16 ./models/CosyVoice3-2512_F16.gguf \
  --out ./models/CosyVoice3-2512_F16_HiFT-F32.gguf
```

成功時會輸出 JSON，確認 `"layers_bit_identical_to_f16": true` 且產出檔案較官方 F16 多約 `65 MB`。

### Stage 1：資料正規化、Token 抽取與防遺忘 Replay Pack

1. **準備角色訓練集**（自動執行靜音修剪至 75 ms 前導／150 ms 尾段淡出、`-20 dBFS` 響度正規化、25 Hz Speech Token 與原生 BPE Token 抽取、防洩漏 `split.json` 及 Top-5 參考音評分 `reference_candidates.json`）：

```bash
python tools/voice_overlay_training/prepare_dataset.py --config train_config.json
```

> 建議人工試聽 `data/processed/reference_candidates.json` 的 Top-5 候選，挑選咬字清晰、語氣自然中性、長約 4.5–7.5 秒且無異常換氣的一句作為正式參考音。

2. **建立可跨角色共用的 Base Replay Pack**（一次性；透過底座自我蒸餾 16 句中／英／情緒指令樣本，防止單語種小語料微調洗壞泛化能力）：

```bash
python tools/voice_overlay_training/build_replay_pack.py \
  --config train_config.json \
  --out-dir ./data/replay \
  --prompt-speech ./data/processed/recommended_reference.gguf
```

（若已有一批乾淨多語種 WAV+TXT，亦可改用 `--from-dir <dir>` 匯入，免去合成步驟。）

### Stages 2–3：純 FP32 LoRA 微調與原生 Overlay 匯出

先跑設定與資料防呆檢查（不會啟動 GPU 訓練）：

```bash
python tools/voice_overlay_training/train_cosyvoice_llm.py \
  --config train_config.json --dry-run-check
```

確認樣本數、切分與參數正確後，加上安全閘門 `--start-training` 開始訓練：

```bash
python tools/voice_overlay_training/train_cosyvoice_llm.py \
  --config train_config.json --start-training
```

#### 推薦超參數總表（經多角色實聽驗證之穩健基線）

| 參數 | 預設推薦值 | 設計依據 |
| :--- | :--- | :--- |
| 底座精度與運算 | 官方 `F32` GGUF + 純 FP32（關 TF32） | 確保低秩梯度無 BF16/F16/TF32 截斷噪聲，單卡僅需 ~5–6 GB VRAM |
| `lora_r` / `lora_alpha` | `16` / `32` | 風格擬合度與小語料防過擬合甜蜜點（產出 F32 Overlay 約 `33.6 MB`） |
| `use_rslora` | `true` | 採 `alpha / sqrt(r)` 縮放，穩定不同 rank 的更新幅度 |
| `lora_plus_ratio` | `2.0` | LoRA+ 讓 `lora_B` 以 2× 學習率收斂，避免初期空轉又不過猛 |
| `lora_dropout` | `0.10` | 小樣本（30–80 句）防死背語句；語料 >120 句可降至 `0.05` |
| `learning_rate` | `2e-5`（30–80 句）<br>`3e-5 ~ 5e-5`（>100 句） | 保護底座節奏與發音，避免早期過衝產生口齒不清或語氣僵硬 |
| `lower_layers_end` / `lower_layers_lr_mult` | `8` / `0.25` | 降低前 8 層（`L0..L7`）更新速率，保護中／日／英底層音素與文字對齊 |
| `train_speech_head` / `llm_decoder_lora_r` | `false` / `0` | 凍結詞表與輸出頭，徹底避免單語種語料破壞跨語種 Token 機率分布 |
| `label_smoothing` / `eos_loss_weight` | `0.05` / `1.5` | 減少過擬合尖峰，並加強 EOS 終止訊號防止尾音拖長或截斷 |
| `zero_shot_pair_prob` / `instruct_prob` | `0.60` / `0.15` | 60% 同角色參考配對（僅對目標段算 loss）＋ 15% 指令模式＋ 25% 單句模式 |
| `replay_ratio` | `0.25` | 每輪混入 25% 底座多語種＋指令回放樣本，同步監控 `val_replay_ce` 漂移 |
| `epochs` / `save_every_epochs` | `5–6`（約 50–80 steps）/ `1` | 每輪輸出 `checkpoints/epoch_XX.gguf` 供 Stage 4 實聽盲測挑檔 |

#### 訓練期四維監控指標（`train_metrics.json`）
- `val_zeroshot_ce`：主選模指標，應隨 epoch 平滑下降。
- `val_single_ce`：無參考單句交叉熵，觀察是否過快下降而拉開與 `val_zeroshot_ce` 的落差。
- `val_replay_ce` 與 `replay_drift`：衡量泛化遺忘程度；健康值通常落在 `<= +0.05`（甚至微負），若持續超過 `+0.15` 表示過擬合或學習率過高。
- `val_eos_loss`：衡量句尾停頓與收尾準確度，應保持穩定不飆升。

### Stage 4：自由生成驗收矩陣與多 Seed 試聽鎖定

> **核心原則：絕對不要只憑最低 validation loss 就直接上線！**
> CosyVoice 小語料 LoRA 常在 loss 仍低時就出現微弱跳字、語氣變平、換氣加重或跨語口音漂移，必須靠自由生成試聽鎖定最佳 `(epoch, scale)`。
>
> **推薦試聽方式（Gemini 4 原生多模態聽審）**：
> Stage 1 原始／清洗後素材抽聽與 Stage 4 驗收矩陣（通常動輒數十到上百條 WAV），**非常推薦直接交給具備原生音訊理解能力的 Gemini 4 試聽**。相較僅靠數值指標或 ASR 文字轉寫，Gemini 4 能直接聽出「逗號處異常大口吸氣、日文漢字／片假名輕微誤讀或拆詞、中英混讀生硬重音、`instruct` 情緒是否真正拉開、以及與 `base` 基線的音色貼合度差異」，大幅降低人工逐條盲聽的負擔；待其篩出 Top 1–2 候選後，再由真人做最終確認即可。

1. **第一輪掃描**（固定 `seed=42`，對比 `base` 與 `epoch_02..05` × `scales = [0.55, 0.70, 0.85]`）：

```bash
python tools/voice_overlay_training/eval_overlay_matrix.py \
  --config train_config.json \
  --overlays ./runs/my_custom_voice/checkpoints/epoch_02.gguf \
             ./runs/my_custom_voice/checkpoints/epoch_03.gguf \
             ./runs/my_custom_voice/checkpoints/epoch_04.gguf \
             ./runs/my_custom_voice/checkpoints/epoch_05.gguf \
  --scales 0.55 0.70 0.85
```

執行後打開 `eval_matrix/listening_report.md` 逐句試聽（涵蓋原語種 `zero-shot`、跨語種 `cross-lingual`、開心／溫柔 `instruct`、長句 EOS 與中英混讀）。

2. **第二輪多隨機種子穩健複驗**（針對第一輪勝出的 1–2 組候選加測多組 seed）：

```bash
python tools/voice_overlay_training/eval_overlay_matrix.py \
  --config train_config.json \
  --overlays ./runs/my_custom_voice/checkpoints/epoch_04.gguf \
  --scales 0.70 \
  --seeds 42 101 202 \
  --skip-base \
  --out-dir ./runs/my_custom_voice/eval_multiseed
```

#### 驗收過關標準（全部達標再鎖定）
- **數值硬門檻**：`EOS Limit == 0`、`Clipped == 0`、無長尾空轉（尾段靜音合理）。
- **聽感門檻**：
  1. 原語種與跨語種均無漏字、重複、拆詞或生硬重音；
  2. 語氣與音色明顯比 `base` 更貼合目標角色；
  3. `instruct`（溫柔／開心等）確實能拉開語氣差異，不會因過擬合而變得扁平；
  4. 長句句間換氣自然，無異常大口喘息或句尾突兀截斷；
  5. 多組隨機種子（`42 / 101 / 202`）皆穩定過關，不靠單一運氣種子。

### Stage 5：在 `cosyvoice.cpp` 中載入與部署

將鎖定的 `epoch_XX.gguf` 重新命名為語意化檔名（如 `my_voice_lora.gguf`），與共用底座 `CosyVoice3-2512_F16_HiFT-F32.gguf` 及驗證通過的參考音 `my_voice_ref.gguf` 搭配使用：

```bash
# CLI 單次測試
cosyvoice-cli \
  --model ./models/CosyVoice3-2512_F16_HiFT-F32.gguf \
  --voice-adapter ./models/adapters/my_voice_lora.gguf \
  --voice-adapter-scale 0.70 \
  --llm-kv-cache-type f16 \
  --prompt-speech ./models/my_voice_ref.gguf \
  --text "你好，今天也一起加油吧。" \
  --output out.wav

# OpenAI 兼容 API 服務
cosyvoice-server \
  --model ./models/CosyVoice3-2512_F16_HiFT-F32.gguf \
  --voice-adapter ./models/adapters/my_voice_lora.gguf \
  --voice-adapter-scale 0.70 \
  --llm-kv-cache-type f16 \
  --voice-prompt default=./models/my_voice_ref.gguf \
  --host 127.0.0.1 --port 8080
```

啟動後可檢查 `/healthz` 回傳的 `voice_adapter_loaded` 欄位確認 Overlay 已成功合併載入。

---

## 4. 常見問題排查（Troubleshooting）

| 現象 | 最常見根因 | 處理方式 |
| :--- | :--- | :--- |
| `instruct` 或長句在逗號處反覆大口喘氣 | 訓練集原音本身含大量近距離句間吸氣，被 LoRA 學入 | 先跑 `debreath_audio.py --dry-run` 檢查；對乾淨副本去吸氣並剔除重喘樣本後重訓，並同步更新參考音 |
| 日文 `zero-shot` 頻繁吃字、拆詞或讀錯漢字 | 訓練文本與參考音 `prompt_speech.gguf` 的文字格式不一致（例如一邊片假名、一邊漢字） | 確認 `raw/*.txt` 與參考音 `prompt_text` 使用相同格式，並確認推論載入的是新生成的 `prompt_speech.gguf` |
| 角色像了，但跨語種口音變怪或情緒指令變平 | 過擬合（epoch 過多、學習率偏高或 `scale` 過大） | 先在 Stage 4 測試較低 `--voice-adapter-scale`（`0.55`）與較早 epoch（`epoch_02` / `03`）；重訓時調高 `replay_ratio` 至 `0.25`、降 `lr` 至 `2e-5` |
| 長句生成不停止、觸發 `EOS_LIMIT` | 訓練集含過多過長尾段靜音或斷尾樣本 | 啟用 Stage 1 `audio_preprocess.trim_silence`（`tail_ms=150`）、保持 `eos_loss_weight=1.5` 並剔除未說完的半句 |
| 載入 Overlay 時失敗或拒絕啟動 | 底座 `layers.*` 為量化型別（如 `Q8_0`），或 Overlay 權重與底座維度不符 | 改用官方 `F16` 或 `build_mixed_base.py` 產出的混合精度底座（`layers.*` 為 `F16` / `F32`） |
