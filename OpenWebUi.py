# dependencies: requests
from typing import List, Union, Generator
from pipelines import Pipeline
import requests

class LLMCodeAnalysisPipeline(Pipeline):
    def __init__(self):
        """تهيئة الأنبوب مع قيم الإعداد."""
        self.type = "pipe"  # نوع الأنبوب
        self.valves = {
            "TASK_MODEL": "llama3.3:70b",              # اسم النموذج
            "API_URL": "http://host.docker.internal:3000/api/chat",  # نقطة نهاية الـ API
            "MAX_TOKENS": 2000                        # الحد الأقصى لعدد الرموز
        }

    async def on_startup(self):
        """تُستدعى عند بدء تشغيل الخادم. يمكن التحقق من اتصال الـ API هنا."""
        pass  # تُركت فارغة حاليًا، يمكن توسعتها لاحقًا

    async def on_shutdown(self):
        """تُستدعى عند إيقاف الخادم. يمكن تنظيف الموارد هنا."""
        pass  # تُركت فارغة حاليًا

    def call_llm(self, prompt: str, code: str) -> str:
        """استدعاء النموذج اللغوي عبر الـ API وإرجاع الإجابة."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.valves["TASK_MODEL"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "أنت خبير في تحليل الأكواد. قم بتحليل كود Python بدقة وأعط إجابات واضحة ومنظمة. "
                        "أجب فقط عن المهمة المطلوبة وتجنب المعلومات غير الضرورية."
                    )
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\n**الكود:**\n```python\n{code}\n```"
                }
            ],
            "max_tokens": self.valves["MAX_TOKENS"]
        }
        try:
            response = requests.post(self.valves["API_URL"], headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"خطأ في استدعاء النموذج: {str(e)}"

    def analyze_structure(self, code: str) -> str:
        """تحليل هيكلية الكود."""
        prompt = (
            "قم بتحليل الكود Python التالي وصف هيكليته. "
            "قم بإدراج جميع الفئات، الدوال، المتغيرات العامة، والاستيرادات المهمة. "
            "قدم الإجابة بشكل واضح ومنظم (مثل قائمة أو أقسام). "
            "مثال:\n- الفئات: الاسم، الوصف\n- الدوال: الاسم، المعاملات\n- المتغيرات العامة: الاسم، النوع"
        )
        return self.call_llm(prompt, code)

    def explain_elements(self, code: str, structure: str) -> str:
        """شرح عناصر الكود بناءً على الهيكلية."""
        prompt = (
            f"استند إلى هيكلية الكود التالية:\n{structure}\n"
            "اشرح كل متغير ودالة في الكود بالتفصيل. لكل متغير، اذكر:\n"
            "- الاسم\n- النوع (مثل int، str)\n- الغرض\n- الاستخدام\n"
            "لكل دالة، اذكر:\n"
            "- الاسم\n- المعاملات\n- القيمة المُعادة\n- الغرض\n- كيفية استخدامها\n"
            "قدم الإجابة بشكل واضح، مثل قائمة أو جدول."
        )
        return self.call_llm(prompt, code)

    def technical_analysis(self, code: str, structure: str, explanations: str) -> str:
        """إجراء تحليل تقني للكود."""
        prompt = (
            f"استند إلى هيكلية الكود التالية:\n{structure}\n"
            f"والشروحات:\n{explanations}\n"
            "قم بإجراء تحليل تقني للكود. قيّم:\n"
            "- القراءة (مثل التسمية، الهيكلية)\n"
            "- الأداء (مثل الكفاءة، قابلية التوسع)\n"
            "- قابلية الخطأ (مثل نقص معالجة الأخطاء)\n"
            "- الجوانب الأمنية (مثل التحقق من المدخلات)\n"
            "قدم اقتراحات تحسين محددة واشرح أهميتها. "
            "نظم الإجابة في أقسام لكل فئة."
        )
        return self.call_llm(prompt, code)

    def professional_analysis(self, code: str, structure: str, explanations: str, tech_analysis: str) -> str:
        """إجراء تحليل احترافي للكود."""
        prompt = (
            f"استند إلى هيكلية الكود التالية:\n{structure}\n"
            f"الشروحات:\n{explanations}\n"
            f"التحليل التقني:\n{tech_analysis}\n"
            "قم بإجراء تحليل احترافي. قيّم:\n"
            "- الصيانة (مثل الوحدات، التوثيق)\n"
            "- قابلية التوسع (مثل ملاءمة المشاريع الكبيرة)\n"
            "- الملاءمة للغرض (هل يحقق الكود المتطلبات؟)\n"
            "- التوافق مع أفضل الممارسات (مثل PEP 8 لـ Python)\n"
            "قدم توصيات لتحسين الكود وقارنه بمعايير الصناعة. "
            "نظم الإجابة في أقسام واضحة."
        )
        return self.call_llm(prompt, code)

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator:
        """
        الدالة الرئيسية للأنبوب. تحلل الكود خطوة بخطوة وتُرجع النتائج كمولد.

        الوسائط:
            user_message (str): الكود المراد تحليله.
            model_id (str): معرف النموذج (غير مستخدم هنا، مضاف للتوافق).
            messages (List[dict]): سجل المحادثة (غير مستخدم هنا).
            body (dict): معلمات إضافية (غير مستخدمة هنا).

        الإرجاع:
            Generator: يُرجع نتائج التحليل خطوة بخطوة.
        """
        code = user_message

        def generate():
            # الخطوة 1: تحليل الهيكلية
            structure = self.analyze_structure(code)
            if "خطأ" in structure:
                yield f"فشل تحليل الهيكلية: {structure}"
                return
            yield "**هيكلية الكود:**\n" + structure + "\n\n"

            # الخطوة 2: الشروحات
            explanations = self.explain_elements(code, structure)
            if "خطأ" in explanations:
                yield f"فشل الشرح: {explanations}"
                return
            yield "**شروحات المتغيرات/الدوال:**\n" + explanations + "\n\n"

            # الخطوة 3: التحليل التقني
            tech_analysis = self.technical_analysis(code, structure, explanations)
            if "خطأ" in tech_analysis:
                yield f"فشل التحليل التقني: {tech_analysis}"
                return
            yield "**التحليل التقني:**\n" + tech_analysis + "\n\n"

            # الخطوة 4: التحليل الاحترافي
            prof_analysis = self.professional_analysis(code, structure, explanations, tech_analysis)
            if "خطأ" in prof_analysis:
                yield f"فشل التحليل الاحترافي: {prof_analysis}"
                return
            yield "**التحليل الاحترافي:**\n" + prof_analysis

        return generate()
