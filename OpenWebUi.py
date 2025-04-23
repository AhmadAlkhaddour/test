---
requirements: requests
---

from pipelines import Pipeline
import requests
import json

class LLMCodeAnalysisPipeline(Pipeline):
    def __init__(self):
        self.type = "pipe"
        self.valves = {
            "TASK_MODEL": "llama3.3:70b",  # LLaMA 3.3 70B Modell
            "API_URL": "http://host.docker.internal:3000/api/chat",  # OpenWebUI API
            "MAX_TOKENS": 2000  # Max. Tokens für LLM-Antworten
        }

    def call_llm(self, prompt, code):
        """Sende Prompt an LLaMA 3.3 70B und hole die Antwort."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.valves["TASK_MODEL"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Du bist ein Experte für Code-Analyse. Analysiere Python-Code präzise und gib klare, strukturierte Antworten. "
                        "Antworte nur auf die gestellte Aufgabe und vermeide unnötige Informationen."
                    )
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\n**Code:**\n```python\n{code}\n```"
                }
            ],
            "max_tokens": self.valves["MAX_TOKENS"]
        }
        try:
            response = requests.post(self.valves["API_URL"], headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Fehler beim LLM-Aufruf: {str(e)}"

    def analyze_structure(self, code):
        """LLM analysiert die Struktur des Codes."""
        prompt = (
            "Analysiere den folgenden Python-Code und beschreibe seine Struktur. "
            "Liste alle Klassen, Funktionen, globale Variablen und wichtige Importe auf. "
            "Gib die Antwort in einer klaren, strukturierten Form (z. B. als Liste oder Abschnitte). "
            "Beispiel:\n- Klassen: Name, Beschreibung\n- Funktionen: Name, Parameter\n- Globale Variablen: Name, Typ"
        )
        return self.call_llm(prompt, code)

    def explain_elements(self, code, structure):
        """LLM erklärt Variablen und Methoden."""
        prompt = (
            f"Basiere auf der folgenden Code-Struktur:\n{structure}\n"
            "Erkläre jede Variable und Methode im Code detailliert. Für jede Variable gib an:\n"
            "- Name\n- Typ (z. B. int, str)\n- Zweck\n- Verwendung\n"
            "Für jede Methode/Funktion gib an:\n"
            "- Name\n- Parameter\n- Rückgabewert\n- Zweck\n- Wie sie verwendet wird\n"
            "Formatiere die Antwort klar, z. B. als Liste oder Tabelle."
        )
        return self.call_llm(prompt, code)

    def technical_analysis(self, code, structure, explanations):
        """LLM führt technische Analyse durch."""
        prompt = (
            f"Basiere auf der folgenden Code-Struktur:\n{structure}\n"
            f"Und den Erklärungen:\n{explanations}\n"
            "Führe eine technische Analyse des Codes durch. Bewerte:\n"
            "- Lesbarkeit (z. B. Benennung, Struktur)\n"
            "- Performance (z. B. Effizienz, Skalierbarkeit)\n"
            "- Fehleranfälligkeit (z. B. fehlende Fehlerbehandlung)\n"
            "- Sicherheitsaspekte (z. B. Eingabevalidierung)\n"
            "Gib konkrete Verbesserungsvorschläge und erkläre, warum sie wichtig sind. "
            "Formatiere die Antwort in Abschnitten für jede Kategorie."
        )
        return self.call_llm(prompt, code)

    def professional_analysis(self, code, structure, explanations, tech_analysis):
        """LLM führt professionelle Analyse durch."""
        prompt = (
            f"Basiere auf der folgenden Code-Struktur:\n{structure}\n"
            f"Erklärungen:\n{explanations}\n"
            f"Technische Analyse:\n{tech_analysis}\n"
            "Führe eine professionelle Analyse durch. Bewerte:\n"
            "- Wartbarkeit (z. B. Modularität, Dokumentation)\n"
            "- Skalierbarkeit (z. B. Eignung für größere Projekte)\n"
            "- Eignung für den Zweck (z. B. Erfüllt der Code die Anforderungen?)\n"
            "- Übereinstimmung mit Best Practices (z. B. PEP 8 für Python)\n"
            "Gib Empfehlungen, wie der Code verbessert werden kann, und vergleiche ihn mit Industriestandards. "
            "Formatiere die Antwort in klaren Abschnitten."
        )
        return self.call_llm(prompt, code)

    def pipe(self, user_message: str, model_id: str, messages: list, body: dict) -> str:
        """Hauptfunktion, die alle Schritte koordiniert."""
        code = user_message  # Eingabe ist der Code

        # Schritt retro: Schritt 1: Strukturanalyse
        structure = self.analyze_structure(code)
        if "Fehler" in structure:
            return f"Strukturanalyse fehlgeschlagen: {structure}"

        # Schritt 2: Variablen/Methoden erklären
        explanations = self.explain_elements(code, structure)
        if "Fehler" in explanations:
            return f"Erklärung fehlgeschlagen: {explanations}"

        # Schritt 3: Technische Analyse
        tech_analysis = self.technical_analysis(code, structure, explanations)
        if "Fehler" in tech_analysis:
            return f"Technische Analyse fehlgeschlagen: {tech_analysis}"

        # Schritt 4: Professionelle Analyse
        prof_analysis = self.professional_analysis(code, structure, explanations, tech_analysis)
        if "Fehler" in prof_analysis:
            return f"Professionelle Analyse fehlgeschlagen: {prof_analysis}"

        # Schritt 5: Bericht zusammenstellen
        return (
            f"**Code-Struktur:**\n{structure}\n\n"
            f"**Erklärungen zu Variablen/Methoden:**\n{explanations}\n\n"
            f"**Technische Analyse:**\n{tech_analysis}\n\n"
            f"**Professionelle Analyse:**\n{prof_analysis}"
        )
