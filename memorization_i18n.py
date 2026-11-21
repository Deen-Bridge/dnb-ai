"""Locale resolution and the deterministic message catalog for the memorization planner.

Contract (#843, AI-067):

* Supported locales are ``en`` and ``ar``. A request may send any BCP-47 tag;
  ``_`` is accepted as a separator and the primary language subtag is matched
  case-insensitively, so ``ar-EG``, ``AR_sa`` and ``ar`` all resolve to ``ar``.
* A missing, malformed or unsupported tag falls back to ``en`` and the response
  says why (``missing`` / ``malformed`` / ``unsupported``). Resolution never
  raises and never calls a model.
* Only user-facing guidance is localized. Quran surah numbers, ayah references,
  hadith citation text and machine-readable codes (``retention_health``,
  ``status``) stay canonical in every locale. The Arabic catalog contains no
  Quran or hadith text: where an English string quotes a hadith, the Arabic
  string refers to the source by collection and number instead.
* English strings are byte-for-byte the strings the planner returned before
  localization, so existing English clients see no change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MESSAGE_CATALOG_VERSION = "2026.11.1"
DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES: tuple[str, ...] = ("en", "ar")
RTL_LOCALES: frozenset[str] = frozenset({"ar"})

# BCP-47 tags are at most 35 characters in practice (RFC 5646 section 4.4.1).
MAX_LOCALE_TAG_LENGTH = 35
_TAG_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*$")

FallbackReason = Literal["missing", "malformed", "unsupported"]


@dataclass(frozen=True)
class LocaleResolution:
    """Outcome of resolving a requested locale tag."""

    requested: str | None
    locale: str
    direction: Literal["ltr", "rtl"]
    fallback: bool
    fallback_reason: FallbackReason | None


def _resolution(requested: str | None, locale: str, reason: FallbackReason | None) -> LocaleResolution:
    return LocaleResolution(
        requested=requested,
        locale=locale,
        direction="rtl" if locale in RTL_LOCALES else "ltr",
        fallback=reason is not None,
        fallback_reason=reason,
    )


def resolve_locale(tag: str | None) -> LocaleResolution:
    """Normalize a BCP-47 tag to a supported planner locale with a safe fallback."""
    if tag is None or not tag.strip():
        return _resolution(None, DEFAULT_LOCALE, "missing")

    normalized = tag.strip().replace("_", "-")
    if len(normalized) > MAX_LOCALE_TAG_LENGTH or not _TAG_PATTERN.fullmatch(normalized):
        # Malformed input is not echoed back.
        return _resolution(None, DEFAULT_LOCALE, "malformed")

    subtags = normalized.split("-")
    primary = subtags[0].lower()
    canonical = "-".join([primary, *subtags[1:]])
    if primary in SUPPORTED_LOCALES:
        return _resolution(canonical, primary, None)
    return _resolution(canonical, DEFAULT_LOCALE, "unsupported")


_CATALOG: dict[str, dict[str, str]] = {
    "en": {
        # Daily schedule tips
        "tip.rest": (
            "Designated rest and spiritual rejuvenation day. Listen to your memorized portion "
            "by a master reciter (e.g., Al-Husary)."
        ),
        "tip.consolidation": (
            "Weekly Consolidation: Recite this week's new verses in voluntary Sunnah prayers to solidify retention."
        ),
        "tip.sabaq": (
            "Sabaq tip: Repeat each ayah at least 15-20 times before linking to the next. "
            "Notice meaning and word endings in Surah {surah}."
        ),
        "tip.review": "Review and consolidate previously memorized portions.",
        "plan.summary": (
            "Personalized schedule for {surahs} surah(s), {ayahs} ayahs ({pages} pages) "
            "at {daily} ayahs/day ({minutes} mins, {days} days/wk)."
        ),
        # Plan milestones
        "milestone.quarter.title": "First Quarter Milestone (25%)",
        "milestone.quarter.description": "Completed 25% of your target memorization goal.",
        "milestone.quarter.message": "A strong start is half the battle. Maintain consistent daily review.",
        "milestone.half.title": "Halfway Milestone (50%)",
        "milestone.half.description": "Alhamdulillah! You have completed half of your memorization journey.",
        "milestone.half.message": "Your commitment has borne fruit. Keep your Sabqi revision rock solid.",
        "milestone.three_quarters.title": "Three-Quarter Milestone (75%)",
        "milestone.three_quarters.description": "75% completed! The finish line is in sight.",
        "milestone.three_quarters.message": "Remain humble, steadfast, and review the beginning portions regularly.",
        "milestone.completion.title": "Goal Completion (100%) - Khatm Milestone",
        "milestone.completion.description": "Mubarak! You have completed your full memorization goal!",
        "milestone.completion.message": (
            "May Allah make the Holy Quran the light of your heart and a guide throughout your life."
        ),
        "milestone.target": "{description} ({mark}/{total} ayahs)",
        # Adaptive adjustment
        "adjust.struggling.health": "Needs Consolidation",
        "adjust.struggling.action": "Slow down new memorization pace by 30% and dedicate 2 days to Sabqi revision.",
        "adjust.struggling.step1": "Pause new Sabaq for the next 48 hours.",
        "adjust.struggling.step2": "Review the last 20 ayahs memorized with a Hafiz or recorded audio repetition.",
        "adjust.struggling.step3": "Recite your recent portions in optional (Nawafil) prayers.",
        "adjust.struggling.encouragement": (
            "Do not be disheartened. In the words of the Prophet ﷺ: 'The one who recites the Quran "
            "with difficulty, stammering over its verses, will receive a double reward.' (Sahih Muslim 798)."
        ),
        "adjust.excelling.health": "Excellent",
        "adjust.excelling.action": "Pace optimal. Opportunity to safely accelerate new memorization by 15%.",
        "adjust.excelling.step1": "Maintain your disciplined daily schedule.",
        "adjust.excelling.step2": "Ensure Manzil cumulative review keeps rotating without neglect.",
        "adjust.excelling.encouragement": (
            "Masha'Allah, your consistency is commendable. 'Continue to review the Quran, "
            "for it slips away faster than untethered camels.' (Sahih al-Bukhari 5033)."
        ),
        "adjust.steady.health": "Steady",
        "adjust.steady.action": "Maintain current balanced pace and revision cycle.",
        "adjust.steady.step1": "Complete today's designated Sabaq portion.",
        "adjust.steady.step2": "Spend at least 15 minutes on recent Sabqi review before sleeping.",
        "adjust.steady.encouragement": "Consistency in small deeds is most beloved to Allah. Keep going steadfastly.",
        # Milestone celebrations
        "celebrate.surah.title": "Completed {surah}!",
        "celebrate.surah.badge": "🏆 Surah Finisher",
        "celebrate.surah.message": (
            "Congratulations on fully memorizing {surah}! What an incredible spiritual achievement."
        ),
        "celebrate.surah.next": (
            "Dedicate the next session to reciting {surah} in its entirety from memory before beginning "
            "the next chapter."
        ),
        "celebrate.surah.fallback_name": "Surah {number}",
        "celebrate.streak.title": "30-Day Unbroken Streak!",
        "celebrate.streak.badge": "🔥 Istiqamah (Steadfastness) Star",
        "celebrate.streak.message": (
            "A full month of daily commitment to the Book of Allah! Your discipline is exemplary."
        ),
        "celebrate.streak.next": "Continue to protect your daily revision time as sacred.",
        "celebrate.complete.title": "Goal Completed! Khatm al-Hifz!",
        "celebrate.complete.badge": "👑 Hafiz Milestone",
        "celebrate.complete.message": "All praise is due to Allah! You have completed your memorization target.",
        "celebrate.complete.next": (
            "Begin the permanent Manzil revision cycle: recite your entire memorized portion weekly or biweekly."
        ),
        "celebrate.half.title": "Halfway Mark Achieved (50%)",
        "celebrate.half.badge": "🌟 Halfway Champion",
        "celebrate.half.message": "You have memorized over half of your designated goal! Keep this momentum.",
        "celebrate.half.next": "Conduct a complete revision of the first half before advancing further.",
        "celebrate.progress.title": "In Progress",
        "celebrate.progress.badge": "🌱 Active Learner",
        "celebrate.progress.message": (
            "You have memorized {count} ayahs so far. Every single letter holds ten rewards."
        ),
        "celebrate.progress.next": "Keep up your daily Sabaq and Sabqi revision.",
        # Muraja'ah reminder
        "revision.reminder": (
            "{due} ayah(s) due for Muraja'ah. Start with {surah}:{ayah} (error rate {error_rate}, overdue {overdue}d)."
        ),
        # Limitations shown with every plan, adjustment and celebration
        "limitation.guidance": (
            "This is scheduling guidance from fixed planner rules, not a religious ruling. "
            "Follow your teacher's guidance for your Hifz."
        ),
        "limitation.estimates": "Dates, durations and paces are estimates; real progress varies by learner.",
        "limitation.heuristics": (
            "Difficulty tiers and page counts are editorial estimates, not taken from a specific printed mushaf."
        ),
    },
    "ar": {
        "tip.rest": "يوم راحة وتجديد للنشاط الروحي. استمع إلى ما حفظته بصوت قارئ متقن (مثل الحصري).",
        "tip.consolidation": (
            "التثبيت الأسبوعي: اقرأ الآيات التي حفظتها هذا الأسبوع في صلوات السنن والنوافل لتثبيت الحفظ."
        ),
        "tip.sabaq": (
            "نصيحة للحفظ الجديد: كرّر كل آية من 15 إلى 20 مرة على الأقل قبل وصلها بالتي تليها. "
            "انتبه إلى المعاني وأواخر الكلمات في سورة {surah}."
        ),
        "tip.review": "راجع ما سبق حفظه وثبّته.",
        "plan.summary": (
            "جدول مخصص لـ {surahs} سورة، {ayahs} آية ({pages} صفحة) "
            "بمعدل {daily} آية يوميًا ({minutes} دقيقة، {days} أيام في الأسبوع)."
        ),
        "milestone.quarter.title": "مرحلة الربع الأول (25%)",
        "milestone.quarter.description": "أتممت 25% من هدف الحفظ.",
        "milestone.quarter.message": "البداية القوية نصف الطريق. حافظ على المراجعة اليومية المنتظمة.",
        "milestone.half.title": "مرحلة منتصف الطريق (50%)",
        "milestone.half.description": "الحمد لله! أتممت نصف رحلة الحفظ.",
        "milestone.half.message": "لقد أثمر التزامك. حافظ على مراجعة المحفوظ القريب (السبقي) بإتقان.",
        "milestone.three_quarters.title": "مرحلة ثلاثة الأرباع (75%)",
        "milestone.three_quarters.description": "أتممت 75%! خط النهاية يلوح في الأفق.",
        "milestone.three_quarters.message": "الزم التواضع والثبات، وراجع المقاطع الأولى بانتظام.",
        "milestone.completion.title": "إتمام الهدف (100%) - مرحلة الختم",
        "milestone.completion.description": "مبارك! أتممت هدف الحفظ كاملًا!",
        "milestone.completion.message": "جعل الله القرآن الكريم نور قلبك وهاديك في حياتك كلها.",
        "milestone.target": "{description} ({mark}/{total} آية)",
        "adjust.struggling.health": "يحتاج إلى تثبيت",
        "adjust.struggling.action": "خفّف وتيرة الحفظ الجديد بنسبة 30% وخصّص يومين لمراجعة المحفوظ القريب (السبقي).",
        "adjust.struggling.step1": "أوقف الحفظ الجديد خلال الثماني والأربعين ساعة القادمة.",
        "adjust.struggling.step2": "راجع آخر 20 آية حفظتها مع حافظ متقن أو بالاستماع المتكرر إلى تسجيل صوتي.",
        "adjust.struggling.step3": "اقرأ محفوظك القريب في صلوات النوافل.",
        "adjust.struggling.encouragement": (
            "لا تحزن؛ فقد ورد في صحيح مسلم (798) أن الذي يقرأ القرآن وهو عليه شاق له أجران."
        ),
        "adjust.excelling.health": "ممتاز",
        "adjust.excelling.action": "وتيرتك مثالية، ويمكنك زيادة الحفظ الجديد بأمان بنسبة 15%.",
        "adjust.excelling.step1": "حافظ على جدولك اليومي المنضبط.",
        "adjust.excelling.step2": "احرص على استمرار دورة المراجعة التراكمية (المنزل) دون انقطاع.",
        "adjust.excelling.encouragement": (
            "ما شاء الله، مواظبتك تستحق الثناء. داوم على تعاهد القرآن؛ فقد ورد في صحيح البخاري (5033) "
            "أنه أسرع تفلتًا من الإبل المعقولة."
        ),
        "adjust.steady.health": "مستقر",
        "adjust.steady.action": "حافظ على وتيرتك المتوازنة ودورة المراجعة الحالية.",
        "adjust.steady.step1": "أتمم مقدار الحفظ الجديد المحدد لليوم.",
        "adjust.steady.step2": "خصّص 15 دقيقة على الأقل لمراجعة المحفوظ القريب (السبقي) قبل النوم.",
        "adjust.steady.encouragement": "المداومة على العمل وإن قلّ من أحب الأعمال إلى الله، فاستمر بثبات.",
        "celebrate.surah.title": "أتممت حفظ سورة {surah}!",
        "celebrate.surah.badge": "🏆 إتمام سورة",
        "celebrate.surah.message": "مبارك عليك إتمام حفظ سورة {surah}! يا له من إنجاز روحي عظيم.",
        "celebrate.surah.next": "خصّص الجلسة القادمة لتلاوة سورة {surah} كاملة من حفظك قبل البدء في السورة التالية.",
        "celebrate.surah.fallback_name": "رقم {number}",
        "celebrate.streak.title": "30 يومًا متتالية دون انقطاع!",
        "celebrate.streak.badge": "🔥 نجمة الاستقامة",
        "celebrate.streak.message": "شهر كامل من الالتزام اليومي بكتاب الله! انضباطك قدوة حسنة.",
        "celebrate.streak.next": "استمر في المحافظة على وقت مراجعتك اليومي ولا تفرّط فيه.",
        "celebrate.complete.title": "اكتمل الهدف! ختم الحفظ!",
        "celebrate.complete.badge": "👑 مرحلة الحافظ",
        "celebrate.complete.message": "الحمد لله! أتممت هدف الحفظ.",
        "celebrate.complete.next": "ابدأ دورة المراجعة الدائمة (المنزل): اقرأ كل محفوظك كل أسبوع أو كل أسبوعين.",
        "celebrate.half.title": "بلغت منتصف الطريق (50%)",
        "celebrate.half.badge": "🌟 بطل منتصف الطريق",
        "celebrate.half.message": "حفظت أكثر من نصف هدفك! حافظ على هذا الحماس.",
        "celebrate.half.next": "راجع النصف الأول مراجعة كاملة قبل أن تتقدم أكثر.",
        "celebrate.progress.title": "قيد التقدم",
        "celebrate.progress.badge": "🌱 متعلم نشط",
        "celebrate.progress.message": "حفظت حتى الآن {count} آية. وفي كل حرف عشر حسنات.",
        "celebrate.progress.next": "واصل حفظك الجديد اليومي ومراجعة المحفوظ القريب.",
        "revision.reminder": (
            "{due} آية مستحقة للمراجعة. ابدأ بالآية {surah}:{ayah} (نسبة الخطأ {error_rate}، التأخر {overdue} يوم)."
        ),
        "limitation.guidance": (
            "هذه إرشادات للجدولة مبنية على قواعد ثابتة في المخطط، وليست حكمًا شرعيًا. اتبع توجيه معلمك في الحفظ."
        ),
        "limitation.estimates": "التواريخ والمدد والوتيرة تقديرية، ويختلف التقدم الفعلي من متعلم لآخر.",
        "limitation.heuristics": "مستويات الصعوبة وعدد الصفحات تقديرات تحريرية، وليست مأخوذة من طبعة مصحف محددة.",
    },
}

LIMITATION_KEYS: tuple[str, ...] = ("limitation.guidance", "limitation.estimates", "limitation.heuristics")


def message(locale: str, key: str, **params: object) -> str:
    """Return the catalog string for ``key`` in ``locale``.

    Unknown locales use the default catalog. A key missing from a non-default
    catalog falls back to English rather than failing a user request; the test
    suite asserts that every catalog has exactly the same keys and placeholders.
    """
    catalog = _CATALOG.get(locale, _CATALOG[DEFAULT_LOCALE])
    template = catalog.get(key) or _CATALOG[DEFAULT_LOCALE][key]
    return template.format(**params)


def limitations(locale: str) -> list[str]:
    return [message(locale, key) for key in LIMITATION_KEYS]


def catalog_keys(locale: str) -> frozenset[str]:
    """Keys defined for ``locale`` (used by tests to keep catalogs in lockstep)."""
    return frozenset(_CATALOG[locale])


def catalog_template(locale: str, key: str) -> str:
    return _CATALOG[locale][key]
