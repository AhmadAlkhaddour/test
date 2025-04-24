# pipeline.py
from typing import List, Generator
import requests
from openwebui.pipelines import Pipeline, pipeline

@pipeline(name="code-analysis", description="LLM-driven Python code analysis")
class LLMCodeAnalysisPipeline(Pipeline):
    def __init__(self):
        """Initialisiert die Pipeline mit Konfigurationswerten."""
        self.type = "pipe"
        self.valves = {
            "TASK_MODEL": "llama3.3:70b",
            "API_URL": "http://host.docker.internal:3000/api/chat",
            "MAX_TOKENS": 2000
        }

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def call_llm(self, prompt: str, code: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.valves["TASK_MODEL"],
            "messages": [
                {"role": "system", "content":
                 "Du bist ein Experte für Code-Analyse. Analysiere Python-Code präzise und gib klare, strukturierte Antworten."},
                {"role": "user", "content": f"{prompt}\n\n**Code:**\n```python\n{code}\n```"}
            ],
            "max_tokens": self.valves["MAX_TOKENS"]
        }
        try:
            resp = requests.post(self.valves["API_URL"], json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Fehler beim Modell-Aufruf: {e}"

    def analyze_structure(self, code: str) -> str:
        prompt = (
            "Analysiere den folgenden Python-Code und beschreibe seine Struktur. "
            "Liste alle Klassen, Funktionen, globalen Variablen und wichtigen Importe auf."
        )
        return self.call_llm(prompt, code)

    def explain_elements(self, code: str, structure: str) -> str:
        prompt = (
            f"Basiere auf dieser Struktur:\n{structure}\n"
            "Erkläre jede Variable und Funktion im Code detailliert."
        )
        return self.call_llm(prompt, code)

    def technical_analysis(self, code: str, structure: str, explanations: str) -> str:
        prompt = (
            f"Struktur:\n{structure}\nErklärungen:\n{explanations}\n"
            "Führe eine technische Analyse durch (Lesbarkeit, Performance, Sicherheit)."
        )
        return self.call_llm(prompt, code)

    def professional_analysis(self, code: str, structure: str, explanations: str, tech_analysis: str) -> str:
        prompt = (
            f"Struktur:\n{structure}\nErklärungen:\n{explanations}\nTechnische Analyse:\n{tech_analysis}\n"
            "Führe eine professionelle Analyse durch (Wartbarkeit, Best Practices)."
        )
        return self.call_llm(prompt, code)

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator:
        code = user_message
        def generate():
            struct = self.analyze_structure(code)
            if "Fehler" in struct:
                yield f"Strukturanalyse fehlgeschlagen: {struct}"; return
            yield "**Code-Struktur:**\n" + struct + "\n\n"

            explains = self.explain_elements(code, struct)
            if "Fehler" in explains:
                yield f"Erklärung fehlgeschlagen: {explains}"; return
            yield "**Erklärungen:**\n" + explains + "\n\n"

            tech = self.technical_analysis(code, struct, explains)
            if "Fehler" in tech:
                yield f"Technische Analyse fehlgeschlagen: {tech}"; return
            yield "**Technische Analyse:**\n" + tech + "\n\n"

            prof = self.professional_analysis(code, struct, explains, tech)
            if "Fehler" in prof:
                yield f"Professionelle Analyse fehlgeschlagen: {prof}"; return
            yield "**Professionelle Analyse:**\n" + prof
        return generate()
