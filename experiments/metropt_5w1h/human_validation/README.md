# MetroPT 人因验证包（Human Validation 数据入口）

> **状态 2026-09-25：评分完成。** 3 位评分者（S01–S03：1 位讲师、
> 1 位研究生、1 位机械工程师，均非作者，自愿参与，每人约 1.5–2 小时）
> 的 JSON 已存于 `ratings_in/`，分析结果位于 `analysis/`。本目录作为
> MetroPT 人工验证材料随复现包一并保存。

> 本目录与主实验输出分离：评分数据、盲评工具和人评分析均保存在
> `human_validation/`，主实验冻结输出仍位于 `../runs/metropt3/`。

## 结果摘要（2026-09-25，论文用数）

- **系统级排序与自动裁判完全一致**（ρ=1.00）：B1 4.03 > B8 3.76 > B2 3.65 > B5 2.55；
- **可读性反转复现**：B8 1.66 vs B1 3.68（p<1e-15）；B2 1.54 与 B8 无显著差异（p=0.41）；
- **state@签名可见正例**：B1/B2 4.29/4.33 > B8 3.54 ≫ B5 1.88；
- 名义对照上 B8 的谨慎对冲被判 state 较低（3.14 vs 4.06），反映选择性校准的残余影响；
- 评分者间一致性：compl 0.84 / act 0.73 / faith 0.26（最弱）/ 可读性 W 0.19；
- 人-裁判报告级 ρ=0.58（n=116）；
- 两处仪器披露：B1 模板 oil-elevation 槽的 `n/a` 伪影出现在 28/30 个样本中（对 B1 不利方向）；
  B8 的 3 份算术错误报告中有 2 份落入样本（人评 faith 2.5 vs 名义均值 2.9）。

主要的论文级分析脚本是：

```text
../src/metropt_human_analysis.py
```

它读取 `ratings_in/` 和 `../runs/metropt3/`，并把 CSV/JSON 结果直接写入
本目录的 `analysis/`。`analysis/analyze_metropt_human.py` 是一个较轻量的、
可独立运行的复核脚本，会额外生成 `human_metropt_summary.json`。

## 目录结构

```text
human_validation/
├── README.md
├── RATER_RECRUITMENT.md
├── instrument/
│   ├── rating_form.html
│   └── sample_manifest.json
├── ratings_in/
│   └── human_validation_metropt_<rater>.json
└── analysis/
    ├── SELECTION_LOG.md
    ├── analyze_metropt_human.py
    ├── human_system_means.csv
    ├── human_stratum_system.csv
    ├── human_readability.csv
    ├── human_inter_rater.csv
    ├── human_vs_judge_reportlevel.csv
    ├── human_tests.json
    └── human_per_rater.csv
```

## 设计要点

- **系统**：B1 模板 / B2 传统 D2T / B5 工具代理 / B8 QCKUR（预注册四系统子集）；
- **实例**：30 窗分层（seed 42）：8 签名可见正例 + 7 无签名正例 + 12 名义对照 + 3 OOC 候选对照；
- **盲评**：每实例 A–D 字母与系统映射独立随机（映射保存在 `instrument/sample_manifest.json`）；
- **维度**：忠实性、完备性、连贯性、可行动性、状态主张正确性、行动建议适当性（各 1–5）+ 可读性排名（1–4）；
- 评分者间一致性和与自动裁判（L3）的相关性均由分析脚本计算。

## 数据流

1. 评分者打开 `instrument/rating_form.html`，填写 rater 代号并完成评分；
2. 下载得到 `human_validation_metropt_<代号>.json`，放入 `ratings_in/`；
3. 从 `metropt_5w1h/` 根目录运行论文级分析：

```bash
python src/metropt_human_analysis.py
```

   或运行轻量复核：

```bash
python human_validation/analysis/analyze_metropt_human.py
```

4. 结果保存在 `human_validation/analysis/`。

## 纪律提醒

- 评分者不得是作者；参与应为自愿，并按研究所采用的伦理/同意程序执行；
- 盲评期间不向评分者披露 `sample_manifest.json` 中的系统映射；
- 完成后的 instrument、ratings 和 analysis 应一起保留，以支持论文的人评结果复核。
