"""VLM Agent: visual validation via Gemini 3.1 Flash Lite."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from PIL import Image

from .few_shot import FewShotGallery
from .math_agent import MathStats, math_opinion_label
from .rate_limiter import RateLimiter

VLM_PROMPT_TEMPLATE = """
У тебя роль эксперта в нейроонкологии и анализе интраоперационных ОКТ-сканов головного мозга.
Перед тобой результаты сегментации ткани, выполненной базовым классификатором SVM, и их математический анализ.

Цветовая кодировка на изображении:
- Зеленый = Здоровая белая ткань (Normal tissue)
- Желтый = Поврежденная ткань (Damaged matter / DM)
- Красный = Опухолевая инфильтрация (Tumor)

{few_shot_section}

ДАННЫЕ МАТЕМАТИЧЕСКОГО АГЕНТА ДЛЯ TEST-СКАНА:
- Процент площади красного цвета (опухоль): {red_percentage}%
- Процент площади желтого цвета (повреждения): {yellow_percentage}%
- Процент площади зеленого цвета (норма): {green_percentage}%
- Коэффициент кластеризации красного цвета: {clustering_ratio}% (высокий процент означает, что красные пиксели собраны в единый крупный очаг; низкий — что они рассеяны как случайный шум).
- Предварительный вердикт математического агента: {math_decision}

ТВОЯ ЗАДАЧА:
Используй эталонные примеры (N1–N12 = Normal, T1–T12 = Tumor) как визуальную «насмотренность».
Сравни TEST-скан с эталонами по морфологии красных зон, глубине прорастания и окружению (зелёный/жёлтый).
Прими финальное решение только для TEST-скана: действительно ли на нём присутствует опухолевая инфильтрация (TUMOR), либо это артефакт/шум сегментации (NOISE).

Учитывай следующие клинические и топические правила:
1. Морфология шума: Если красные пиксели представляют собой тонкие, изолированные вертикальные полосы, окруженные преимущественно зеленым цветом, — это с высокой вероятностью шум (FPR модели SVM).
2. Морфология опухоли: Истинная опухоль обычно выглядит как плотные скопления красного цвета, которые прорастают вертикально вглубь (к нижней границе скана) и часто граничат с желтой зоной повреждения.

Выведи результат строго в следующем формате:
[FINAL_DECISION]: <TUMOR или NOISE>
[CONFIDENCE]: <Уверенность от 0% до 100%>
[EXPLANATION]: <Краткое обоснование твоего решения на основе сравнения TEST с эталонами N/T и анализа формы, глубины и окружения красных зон>
"""

FEW_SHOT_SECTION = """
FEW-SHOT ПАНЕЛЬ 5x5:
На приложенном изображении — сетка 5×5 (25 ячеек):
  • N1–N12 (зелёная рамка) — эталоны Normal: SVM дал ложноположительные красные зоны, но опухоли нет.
  • T1–T12 (красная рамка) — эталоны Tumor: подтверждённая опухолевая инфильтрация.
  • TEST (толстая красная рамка, центр сетки) — скан, который нужно классифицировать.

Расположение ячеек (слева направо, сверху вниз):
{layout_description}
"""


@dataclass
class VLMResult:
    final_decision: str
    confidence: float
    explanation: str
    raw_response: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VLMAgent:
    DEFAULT_MODEL = "gemini-3.1-flash-lite"
    DEFAULT_MAX_REQUESTS_PER_MINUTE = 10

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_requests_per_minute: int | None = None,
        few_shot_gallery: FewShotGallery | None = None,
        use_few_shot: bool = True,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        self.model_name = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL)
        self.temperature = temperature
        self.use_few_shot = use_few_shot
        self.few_shot_gallery = few_shot_gallery
        rpm = max_requests_per_minute or int(
            os.getenv("GEMINI_MAX_RPM", self.DEFAULT_MAX_REQUESTS_PER_MINUTE)
        )
        self._rate_limiter = RateLimiter(max_calls=rpm, period_seconds=60.0)
        self._model = self._init_model()

    def _init_model(self):
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generation_config = genai.GenerationConfig(temperature=self.temperature)

        return genai.GenerativeModel(
            self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    def analyze(self, image_path: str | Path, math_stats: MathStats) -> VLMResult:
        self._rate_limiter.wait()

        image_path = Path(image_path)
        few_shot_panel = None
        if self.use_few_shot and self.few_shot_gallery is not None:
            few_shot_panel = self.few_shot_gallery.build_panel(image_path, math_stats)

        prompt = self._build_prompt(math_stats, few_shot_panel)
        contents: list[Any] = [prompt]

        if few_shot_panel is not None:
            contents.append(few_shot_panel.panel_image)
        else:
            contents.append(Image.open(image_path).convert("RGB"))

        response = self._model.generate_content(contents)
        raw_text = response.text or ""

        return VLMResult(
            final_decision=self._parse_field(raw_text, "FINAL_DECISION", default="NOISE"),
            confidence=self._parse_confidence(raw_text),
            explanation=self._parse_field(raw_text, "EXPLANATION", default=""),
            raw_response=raw_text,
            model=self.model_name,
        )

    def _build_prompt(self, math_stats: MathStats, few_shot_panel=None) -> str:
        if few_shot_panel is not None:
            few_shot_section = FEW_SHOT_SECTION.format(
                layout_description=few_shot_panel.layout_description
            )
        else:
            few_shot_section = (
                "Перед тобой один TEST-скан (без few-shot панели). "
                "Проанализируй его визуально и сопоставь с математическими расчетами."
            )

        return VLM_PROMPT_TEMPLATE.format(
            few_shot_section=few_shot_section,
            red_percentage=math_stats.red_percentage,
            yellow_percentage=math_stats.yellow_percentage,
            green_percentage=math_stats.green_percentage,
            clustering_ratio=math_stats.clustering_ratio,
            math_decision=math_opinion_label(math_stats.math_decision),
        ).strip()

    def _parse_field(self, text: str, field: str, default: str = "") -> str:
        pattern = rf"\[{field}\]:\s*(.+?)(?=\n\[|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return default
        return match.group(1).strip()

    def _parse_confidence(self, text: str) -> float:
        value = self._parse_field(text, "CONFIDENCE", default="0%")
        numbers = re.findall(r"[\d.]+", value)
        if not numbers:
            return 0.0
        return min(100.0, max(0.0, float(numbers[0])))
