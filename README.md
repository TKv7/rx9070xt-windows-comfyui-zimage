# RX 9070 XT + Windows + ComfyUI + Z-Image Optimization Notes

Personal benchmark / optimization notes for running Z-Image on **Windows with an AMD Radeon RX 9070 XT** using ComfyUI and ROCm.

> This is a record from one specific machine and workflow, not a universal tuning guide. Results may vary by driver, ComfyUI version, model, workflow and system memory. I may not be able to provide technical support or answer questions.

日本語版は下にあります。 / Japanese version is below.

---

## English

### Goal

The starting point was roughly **6 seconds per 1024×1024 image at 8 steps**. The goal was to reduce end-to-end generation time on Windows without switching back to Linux.

After testing attention backends, VRAM modes, async offload, power limits and targeted model unloading before VAE decode, the final clean result was:

- **1024×1024**
- **8 steps**
- Z-Image Turbo workflow
- INT8 / mixed precision setup
- **5 measured runs: 4.30 / 4.38 / 4.36 / 4.35 / 4.37 s**
- **Average: 4.352 s**

The first run after launch was treated as warm-up and excluded.

### Test environment

- GPU: **AMD Radeon RX 9070 XT 16 GB**
- GPU arch reported by PyTorch: **gfx1201**
- OS: **Windows 11**
- ComfyUI: **0.34.0**
- ComfyUI revision used during testing: `bbbf1f21`
- PyTorch: `2.13.0+rocm10.1.0a20260822`
- ROCm runtime log reported: `(7, 16)`
- comfy-kitchen: **0.2.32**
- comfy-aimdo: **0.5.1**
- VRAM mode: **NORMAL_VRAM / DynamicVRAM**
- Async weight offloading: **enabled, 2 streams**
- Pinned memory: about **12.7 GiB**
- Attention: **PyTorch attention + Comfy Kitchen attention**
- Triton backend: **disabled**
- VAE: bf16
- AMD Adrenalin Power Limit for final fastest result: **0%**

Stable launch flags:

```text
--use-ck-attention --disable-triton-backend --extra-model-paths-config <your_path>
```

### What was tested

#### Triton backend

Enabling Triton caused a crash on this Windows / RX 9070 XT setup with an error similar to:

```text
couldn't allocate input reg for constraint 'r'
```

Result: **Triton was kept disabled.**

#### Disabling async offload

With async offload disabled, repeated runs were around **4.95–4.99 s**.

Result: **worse than the default 2-stream async offload configuration.**

#### HighVRAM mode

HighVRAM could make the sampler itself faster (around **3.65 s** in one test), but VAE decode became much slower because of memory pressure.

Observed behavior included:

- VAE around **1.26–5.7+ s** depending on the exact setup
- without the custom unload logic, VAE could reach roughly **6–10 s**
- total time could exceed **10 s**

Result: **NORMAL_VRAM / DynamicVRAM was much better end-to-end.**

#### Removing the targeted unload tweak

Without the targeted unload before VAE decode in NORMAL_VRAM, the sampler could still be fast (about **3.7 s**), but VAE decode jumped to roughly **5.3–5.5 s**, producing total times around **9.4–9.6 s**.

This was the main clue that the bottleneck was not only the sampler itself, but also the VRAM state immediately before VAE decode.

### Targeted VBAR unload tuning

The key experiment was to partially unload the `Lumina2` model just before VAE decode, while fully unloading `ZImageTEModel_`.

| Lumina2 unload | Sampler avg | VAE avg | Total avg | Notes |
|---:|---:|---:|---:|---|
| 4352 MiB | ~3.51 s | ~0.62 s | ~4.46 s | VAE still slightly slower |
| **4416 MiB** | ~3.50 s | ~0.56 s | **~4.39 s instrumented** | Best balance |
| 4480 MiB | ~3.51 s | ~0.56 s | ~4.39 s | Essentially tied with 4416 |
| 4544 MiB | ~3.55 s | ~0.56 s | ~4.44 s | Sampler became slower |

The difference between 4416 and 4480 MiB was only about **0.002 s**, which is measurement noise. I chose **4416 MiB** because it achieved the same practical performance while unloading 64 MiB less.

After removing benchmark instrumentation overhead, the final clean 4416 MiB result became:

```text
4.30 s
4.38 s
4.36 s
4.35 s
4.37 s
Average: 4.352 s
```

### Final core tweak

Inserted in the `VAEDecode` path before `vae.decode(latent)`:

```python
for _lm in list(comfy.model_management.current_loaded_models):
    _m = _lm.model
    if _m is None:
        continue

    _name = getattr(
        getattr(_m, "model", None),
        "__class__",
        type(None)
    ).__name__

    if _name == "ZImageTEModel_":
        comfy.model_management.unload_model_and_clones(
            _m, unload_additional_models=False
        )

    elif _name == "Lumina2":
        _m.partially_unload(None, 4416 * 1024**2)

comfy.model_management.soft_empty_cache()
images = vae.decode(latent)
```

### Important caveats

- This modifies a **ComfyUI core file**, so an update may overwrite it.
- **4416 MiB is not a universal RX 9070 XT setting.** It is the best value found for this machine, this workflow and this software stack.
- The model-specific unload branches only act on `ZImageTEModel_` and `Lumina2`, but this version still calls `soft_empty_cache()` before every VAE decode. Test unrelated workflows separately.
- The first generation after launch is much slower because of model staging, allocation and kernel/cache initialization. Benchmark from the **second run onward**.

### Power-limit observations

Approximate end-to-end averages observed during tuning:

- PL 0%: **4.35 s** final clean result
- PL -10%: about **4.61 s**
- PL -20%: about **4.73 s**
- PL -30%: about **4.77 s**

These were not all measured with exactly the same intermediate VBAR tuning, so treat them as directional rather than a strict controlled benchmark.

### Practical conclusion

For this specific setup, the best combination was:

- NORMAL_VRAM / DynamicVRAM
- default async offload (2 streams)
- Comfy Kitchen attention
- Triton disabled
- PL 0%
- unload `ZImageTEModel_` before VAE
- partially unload `Lumina2` by **4416 MiB** before VAE

The main lesson was that **optimizing only sampler speed was misleading**. A configuration could make sampling faster but still lose badly overall if VAE decode entered a bad VRAM state.

---

## 日本語

### 目的

Windows上の **Radeon RX 9070 XT + ComfyUI + ROCm** でZ-Imageを動かし、Linuxへ戻さずにどこまで生成時間を詰められるかを検証した記録です。

スタート時は、1024×1024 / 8 stepsでおおむね**6秒前後**でした。

Attention、VRAMモード、async offload、Power Limit、VAE直前のモデルアンロード量を順番に切り分けた結果、最終的には次のところまで到達しました。

- **1024×1024**
- **8 steps**
- Z-Image Turbo系ワークフロー
- INT8 / mixed precision
- 5回実測: **4.30 / 4.38 / 4.36 / 4.35 / 4.37秒**
- **平均 4.352秒**

起動後1回目はウォームアップとして除外しています。

### 検証環境

- GPU: **AMD Radeon RX 9070 XT 16GB**
- GPU arch: **gfx1201**
- OS: **Windows 11**
- ComfyUI: **0.34.0**
- 検証時revision: `bbbf1f21`
- PyTorch: `2.13.0+rocm10.1.0a20260822`
- 起動ログ上のROCm表示: `(7, 16)`
- comfy-kitchen: **0.2.32**
- comfy-aimdo: **0.5.1**
- VRAMモード: **NORMAL_VRAM / DynamicVRAM**
- async weight offloading: **有効 / 2 streams**
- pinned memory: 約 **12.7GiB**
- Attention: **PyTorch attention + Comfy Kitchen attention**
- Triton backend: **無効**
- VAE: bf16
- 最終最速時のAMD Adrenalin Power Limit: **0%**

安定して使った起動フラグ:

```text
--use-ck-attention --disable-triton-backend --extra-model-paths-config <your_path>
```

### 試したこと

#### Triton

Tritonを有効にすると、このWindows + RX 9070 XT環境では次のようなエラーでクラッシュしました。

```text
couldn't allocate input reg for constraint 'r'
```

そのため、**Tritonは無効のまま**にしました。

#### async offload無効化

`--disable-async-offload` を試したところ、安定後の総時間はおおむね**4.95～4.99秒**でした。

最終構成より遅かったため、**標準の2-stream async offloadを維持**しました。

#### HighVRAM

HighVRAMではSampler単体は約**3.65秒**まで速くなる場面がありましたが、VAE Decodeが大きく悪化しました。

条件によっては、

- VAE: **1.26～5.7秒以上**
- カスタムアンロードなしではVAEが**6～10秒程度**
- 総時間が**10秒超**

になることもありました。

結論として、**NORMAL_VRAM / DynamicVRAMの方が良好**でした。

#### VAE直前のアンロード処理を外す

NORMAL_VRAMでVAE直前の調整を完全に外すと、Samplerは約**3.7秒**でも、VAEが約**5.3～5.5秒**まで悪化し、総時間は**9.4～9.6秒前後**になりました。

ここから、Samplerだけではなく、**VAE Decodeに入る直前のVRAM状態が非常に重要**だと判断しました。

### Lumina2の部分アンロード量を調整

VAE直前に、

- `ZImageTEModel_` はアンロード
- `Lumina2` はVBARから一定量だけ部分アンロード

する方向で調整しました。

| Lumina2アンロード量 | Sampler平均 | VAE平均 | 総時間平均 | 傾向 |
|---:|---:|---:|---:|---|
| 4352 MiB | 約3.51秒 | 約0.62秒 | 約4.46秒 | VAEがやや遅い |
| **4416 MiB** | 約3.50秒 | 約0.56秒 | **約4.39秒（計測コードあり）** | バランス最良 |
| 4480 MiB | 約3.51秒 | 約0.56秒 | 約4.39秒 | 4416と実質同等 |
| 4544 MiB | 約3.55秒 | 約0.56秒 | 約4.44秒 | Samplerが悪化 |

4416と4480MiBの差は約**0.002秒**しかなく、誤差範囲と判断しました。

そのため、同じ性能ならより少ないアンロード量で済む**4416MiB**を採用しました。

計測用タイマーなどを削除した最終クリーン状態では、

```text
4.30秒
4.38秒
4.36秒
4.35秒
4.37秒
平均 4.352秒
```

となりました。

### 最終的に入れた処理

`VAEDecode` で `vae.decode(latent)` の直前に次の処理を入れています。

```python
for _lm in list(comfy.model_management.current_loaded_models):
    _m = _lm.model
    if _m is None:
        continue

    _name = getattr(
        getattr(_m, "model", None),
        "__class__",
        type(None)
    ).__name__

    if _name == "ZImageTEModel_":
        comfy.model_management.unload_model_and_clones(
            _m, unload_additional_models=False
        )

    elif _name == "Lumina2":
        _m.partially_unload(None, 4416 * 1024**2)

comfy.model_management.soft_empty_cache()
images = vae.decode(latent)
```

### 注意点

- これは**ComfyUI本体のファイルを直接変更する方法**です。アップデートで上書きされる可能性があります。
- **4416MiBは9070 XT全体の正解ではありません。** このPC、このワークフロー、この時点のソフトウェア構成での最適値です。
- モデル名によるアンロード自体はZ-Image関連だけに反応しますが、この版では `soft_empty_cache()` は毎回実行されます。別ワークフローへ流用する場合は要検証です。
- 起動後1枚目はモデル配置・メモリアロケーション・カーネル/キャッシュ初期化などで遅くなります。**2枚目以降を計測対象**にしています。

### Power Limitについて

途中検証のおおよその総時間:

- PL 0%: **4.35秒**（最終クリーン構成）
- PL -10%: 約**4.61秒**
- PL -20%: 約**4.73秒**
- PL -30%: 約**4.77秒**

ただし、途中のPL比較ではVBARアンロード量などが完全に同一ではない回も含むため、厳密な同条件比較ではなく傾向を見るための数字です。

### 最終結論

この環境では、

- NORMAL_VRAM / DynamicVRAM
- async offload 2 streamsのまま
- Comfy Kitchen attention
- Triton無効
- PL 0%
- VAE前に `ZImageTEModel_` をアンロード
- VAE前に `Lumina2` を **4416MiB** 部分アンロード

という組み合わせが最も良い結果でした。

今回いちばん大きかったのは、**Sampler単体を速くするだけでは意味がない**という点でした。Samplerが速くなっても、その代わりにVAE Decodeが悪化すると総時間では負けます。

---

## Support / サポートについて

This repository is primarily a **personal benchmark and optimization record**. It is published in case the data helps other users with similar hardware. I may not be able to provide technical support or answer questions.

このリポジトリは、基本的に**個人環境での検証記録**です。同じような環境の人の参考になればと思い公開しています。サポート目的ではないため、質問には回答できない場合があります。
