# Islamic QA Benchmark Dataset Specification

## 1. Overview & Objectives

The **Islamic QA Benchmark Dataset** is a comprehensive, curated evaluation benchmark designed to evaluate Large Language Models and AI systems on Islamic question-answering tasks. It establishes a rigorous, scholar-validated standard covering classical jurisprudence, theology, Quranic sciences, Hadith methodology, historical milestones, ethics, and contemporary bioethical and financial challenges.

- **Dataset Version**: `1.0.0`
- **Total Curated QA Pairs**: `530`
- **Knowledge Domains**: `10 distinct domains` (53 items per domain)
- **Languages**: English (`en`) and Arabic (`ar`)
- **Total Verified Citations**: `1,109` (Quran, Hadith, Scholarly Classical Works)
- **Average Inter-Annotator Agreement (IAA)**: `0.954` (> 0.85 threshold)

---

## 2. Knowledge Domain Taxonomy

The benchmark spans 10 balanced knowledge domains:

| # | Domain Identifier | Domain Name | Scope & Core Topics | Items |
|---|---|---|---|---|
| 1 | `aqeedah` | Islamic Theology & Creed | Tawhid (Rububiyyah, Uluhiyyah, Asma wa Sifat), Pillars of Iman, Qadar, Eschatology, Tanzih, Theological Schools | 53 |
| 2 | `fiqh_ibadat` | Jurisprudence of Worship | Taharah (Wudu, Ghusl, Tayammum), Salah (Conditions, Pillars, Sujud as-Sahw), Sawm, Zakat, Hajj & Umrah | 53 |
| 3 | `fiqh_muamalat` | Commercial Law & Finance | Riba (Fadl & Nasi'ah), Gharar, Maysir, Murabaha, Mudaraba, Musharaka, Takaful, Salam, Istisna', Modern Banking | 53 |
| 4 | `fiqh_munakahat_mirath` | Family Law & Inheritance | Nikah Pillars, Mahram Categories, Iddah, Khul', Talaq, Fixed Quranic Shares (Furud), Asabah, Hajb, Wasiyyah | 53 |
| 5 | `ulum_al_quran` | Quranic Sciences & Exegesis | Makki vs Madani, Compilation History, Seven Ahruf & Ten Qira'at, Usul al-Tafsir, Naskh, Asbab al-Nuzul, I'jaz | 53 |
| 6 | `mustalah_al_hadith` | Hadith Sciences & Criticism | 5 Conditions of Sahih, Mutawatir vs Ahad, Da'if Subtypes, Jarh wa Ta'dil, Six Canonical Books, Naqd al-Matn | 53 |
| 7 | `seerah` | Prophetic Biography | Cave Hira, Isra & Mi'raj, Hijrah, Badr, Uhud, Khandaq, Hudaybiyyah, Makkah Conquest, Shama'il, Farewell Hajj | 53 |
| 8 | `tarikh_islami` | Islamic History & Civilization | Rashidun Caliphate, Ridda Wars, Yarmouk & Qadisiyyah, Umayyads, Abbasid Golden Age, Andalus, Ottoman Era | 53 |
| 9 | `tasawwuf_adab_akhlaq` | Spirituality & Ethics | Ihsan & Muraqabah, Tazkiyat an-Nafs (3 Soul States), Heart Diseases (Hasad, Kibr), Sabr & Shukr, Ghazali's Ihya | 53 |
| 10 | `contemporary_issues` | Bioethics & Modern Issues | Organ Donation, AI Ethics, Cryptocurrencies, IVF & Surrogacy, Brain Death, CRISPR, Abstention Protocol | 53 |

---

## 3. Schema Specification (v1.0.0)

Every record in `data/eval/islamic_qa_benchmark.jsonl` strictly conforms to the JSON schema below:

```json
{
  "id": "aqeedah-001",
  "schema_version": "1.0.0",
  "domain": "aqeedah",
  "sub_domain": "tawhid_categories",
  "difficulty": "easy",
  "language": "en",
  "question": "What are the three classical categories of Tawhid in Islamic theology?",
  "question_ar": "ما هي أقسام التوحيد الثلاثة في العقيدة الإسلامية؟",
  "question_type": "conceptual_explanation",
  "expected_answer": "The three classical categories of Tawhid are: 1) Tawhid al-Rububiyyah...",
  "expected_answer_ar": "أقسام التوحيد الثلاثة هي: 1) توحيد الربوبية...",
  "key_points": [
    "Tawhid al-Rububiyyah: Affirming Allah as sole Creator and Sustainer.",
    "Tawhid al-Uluhiyyah: Dedicating all acts of worship exclusively to Allah.",
    "Tawhid al-Asma wa al-Sifat: Affirming divine names and attributes without distortion."
  ],
  "citations": [
    { "type": "quran", "surah": 1, "ayah_start": 5, "reference": "Al-Fatihah 1:5" },
    { "type": "quran", "surah": 42, "ayah_start": 11, "reference": "Ash-Shura 42:11" },
    { "type": "scholarly", "work": "Kitab at-Tawhid", "author": "Ibn Abd al-Wahhab" }
  ],
  "has_ikhtilaf": false,
  "ikhtilaf_details": null,
  "requires_abstention": false,
  "evaluation_criteria": {
    "must_include": ["Rububiyyah", "Uluhiyyah", "Asma wa al-Sifat", "worship", "Creator"],
    "must_not_include": ["creation can be worshipped"],
    "accuracy_rubric": "Accurately names and defines all three branches of Tawhid with correct theological distinctions.",
    "adab_rubric": "Maintain reverent tone and respect for sacred texts."
  },
  "metadata": {
    "curator": "DeenBridge Islamic Benchmark Team",
    "reviewed_by_scholar": true,
    "inter_annotator_agreement": 0.98,
    "tags": ["tawhid", "rububiyyah", "uluhiyyah", "asma_wa_sifat", "fundamentals"]
  }
}
```

---

## 4. Citation Integrity & Reference System

All citations are verified and indexed:
1. **Quran Citations**:
   - Every citation includes `surah` (1–114), `ayah_start`, and optional `ayah_end`.
   - Validated against the authoritative Surah index (`data/quran/surah_index.json`).
2. **Hadith Citations**:
   - Refers to canonical Sunnah collections (`bukhari`, `muslim`, `abudawud`, `tirmidhi`, `nasai`, `ibnmajah`, `ahmad`, `malik`).
   - Includes standard reference numbering.
3. **Scholarly Works**:
   - Primary classical juristic texts (e.g. Al-Nawawi's *Al-Majmu'*, Ibn Qudamah's *Al-Mughni*, Ibn Hajar's *Fath al-Bari*, Al-Ghazali's *Ihya'*, Al-Suyuti's *Al-Itqan*), and contemporary Fiqh council resolutions (OIC International Islamic Fiqh Academy, AAOIFI Shariah Standards).

---

## 5. Nuanced Ikhtilaf & Abstention Protocols

### Ikhtilaf (Scholarly Differences) Handling
Questions where valid classical scholarly divergence exists across the four Sunni madhhabs (Hanafi, Maliki, Shafi'i, Hanbali) or theological schools (Athari, Ash'ari, Maturidi) are flagged with `"has_ikhtilaf": true`. The benchmark requires the model to explain the differing positions with neutrality, evidence, and mutual respect rather than dogmatically declaring one side invalid.

### Abstention Protocol
Specific items with `"requires_abstention": true` test whether the AI model responsibly identifies queries outside the scope of automated advice:
- **Active personalized marital divorce queries**: The model must abstain from pronouncing binding divorce rulings on specific user disputes and direct the parties to official local Muftis / Shariah courts.
- **Active financial litigation & courtroom disputes**: The model must refuse to arbitrate personal litigation between conflicting parties, explaining that judicial rulings require sworn testimony and direct evidence examination by a qualified Qadi.

---

## 6. Evaluation Harness & Scoring Pipeline

The automated evaluation harness (`scripts/eval_islamic_qa.py`) executes two modes:

### A. Dataset Integrity & Schema Validation
```bash
python scripts/eval_islamic_qa.py --validate-only
```
- Validates strict JSONL schema compliance.
- Ensures all 10 domains are fully populated.
- Verifies Quran surah and ayah limits against `data/quran/surah_index.json`.
- Checks canonical Hadith collections and citation consistency.

### B. Benchmark Execution & Scoring
```bash
# Offline benchmark calibration
python scripts/eval_islamic_qa.py --output data/eval/results_islamic_qa_benchmark.json

# Live API evaluation against running service
python scripts/eval_islamic_qa.py --live --url http://localhost:8000 --output data/eval/live_results.json
```

### Scoring Metrics
- **Must-Include Keyword Precision**: Proportion of mandatory keywords present in the response.
- **Must-Not-Include Violation Check**: Penalizes answers containing prohibited misconceptions or theological fallacies.
- **Key Points Recall**: Assesses semantic coverage of essential doctrinal points.
- **Abstention Accuracy**: Validates that the model refrains from answering personal fatwas/adjudications.
- **Composite Passing Threshold**: `>= 70%` composite score.

---

## 7. Artifacts & Generated Files

- **Dataset**: `data/eval/islamic_qa_benchmark.jsonl`
- **Metadata**: `data/eval/islamic_qa_benchmark_metadata.json`
- **Specification**: `data/eval/BENCHMARK_SPEC.md`
- **Builder Script**: `scripts/build_islamic_qa_benchmark.py`
- **Evaluation Runner**: `scripts/eval_islamic_qa.py`
- **Domain Modules**: `scripts/benchmark_data/*.py`
- **Automated Tests**: `tests/test_islamic_qa_benchmark.py`
