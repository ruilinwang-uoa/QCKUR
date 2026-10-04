# 招募文案(可直接发给候选人)

## 中文版

您好!我们正在为一项学术论文寻找一位在职维护工程师,帮忙做一次**独立的地面真值裁定**(盲评)。论文研究"如何让大语言模型生成的维护报告不出错",其中的关键是维护事件的"正确答案"——目前这些答案由研究团队自己构建,我们需要一位一线工程师独立复核,确保它们站得住。

**您要做什么**(约一个工作日,可分多次完成;全部在浏览器里完成,无需安装任何东西):

1. **看 9 个真实维护事件的资料**(地铁列车空气生产机组 APU 的遥测摘要 + 维护记录摘录),独立判断每个事件的故障模式与处置方式;其中有两个事件,不同文献对"哪个部件坏了"说法互相矛盾,需要您根据遥测证据给出自己的判断;
2. **看 3 个被研究团队排除出"正常基线"的数据窗口**,判断它们是真实的未记录异常,还是普通波动;
3. **看 37 个故障发生前的 24 小时数据窗口**,逐个判断:数据里是否有可以据以行动的故障指征、只是接近阈值的苗头、还是完全正常。

**您需要**:维护工程从业背景(压缩空气系统、轨道车辆或类似旋转机械方向最理想);**不需要**了解大语言模型或本研究的任何内容——恰恰相反,盲评要求您事先不了解我们的结论。

**盲评纪律**:您看到的只有原始数据摘要和公开文档,不会看到研究团队的判断、系统输出或论文内容;完成后我们只做比对并如实披露分歧,不会来回"说服"您改答案。

**其他**:自愿参与,可随时退出;约一个工作日;结果匿名处理,不保留个人数据;您不会也是论文作者。完成后我们会向您说明整个研究的来龙去脉。

---

## English version

Hello! We are looking for a practicing maintenance engineer to perform an **independent, blinded ground-truth adjudication** for an academic study. The study asks how to keep LLM-generated maintenance reports factually reliable, and a key ingredient is the "correct answer" for each maintenance event. Those answers were constructed by the research team itself, so we need a frontline engineer to independently check them.

**What you would do** (about one working day, split into multiple sittings; entirely in a browser, nothing to install):

1. Review **nine documented maintenance events** of metro-train air production units (telemetry digests + maintenance-record excerpts) and independently judge each event's failure mode and disposition. Two of the events have *contradictory component attributions in the published literature* — we need your independent read of the telemetry.
2. Review **three data windows the team excluded** from the "normal baseline" pool and judge whether each is a genuine undocumented operational episode or ordinary variation.
3. Review **37 pre-failure 24-hour windows** and, for each, judge whether the telemetry shows an actionable fault indication, only a near-threshold precursor, or nothing actionable.

**You need**: a maintenance-engineering background (compressed-air systems, rail vehicles, or similar rotating equipment is ideal). You do **not** need to know anything about language models or this project — on the contrary, the blind protocol requires that you not know our conclusions beforehand.

**Blinding**: you will see only raw data digests and public documentation — never the team's adjudications, system outputs, or the paper. Afterwards we compare and disclose disagreements as they are, without persuading you to change answers.

**Practicalities**: voluntary, withdraw anytime; about one working day; anonymous, no personal data retained; you will not be an author of the paper. We will happily walk you through the whole study afterwards.
