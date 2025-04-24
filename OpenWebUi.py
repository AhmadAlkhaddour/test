from typing import List, Union, Generator
import requests

class LLMCodeAnalysisPipeline:
    def __init__(self):
        """Initialisiert die Pipeline mit Konfigurationswerten."""
        self.type = "pipe"  # Typ der Pipeline
        self.valves = {
            "TASK_MODEL": "llama3.3:70b",              # Modellname
            "API_URL": "http://host.docker.internal:3000/api/chat",  # API-Endpunkt
            "MAX_TOKENS": 2000                        # Maximale Token-Anzahl
        }

    async def on_startup(self):
        """Wird beim Serverstart aufgerufen. Hier könnte eine API-Verbindung geprüft werden."""
        pass  # Für jetzt leer gelassen, kann später erweitert werden

    async def on_shutdown(self):
        """Wird beim Serverstopp aufgerufen. Hier könnten Ressourcen bereinigt werden."""
        pass  # Für jetzt leer gelassen

    def call_llm(self, prompt: str, code: str) -> str:
        """Ruft das LLM über die API auf und gibt die Antwort zurück."""
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

    def analyze_structure(self, code: str) -> str:
        """Analysiert die Struktur des Codes."""
        prompt = (
            "Analysiere den folgenden Python-Code und beschreibe seine Struktur. "
            "Liste alle Klassen, Funktionen, globale Variablen und wichtige Importe auf. "
            "Gib die Antwort in einer klaren, strukturierten Form (z. B. als Liste oder Abschnitte). "
            "Beispiel:\n- Klassen: Name, Beschreibung\n- Funktionen: Name, Parameter\n- Globale Variablen: Name, Typ"
        )
        return self.call_llm(prompt, code)

    def explain_elements(self, code: str, structure: str) -> str:
        """Erklärt die Elemente des Codes basierend auf der Struktur."""
        prompt = (
            f"Basiere auf der folgenden Code-Struktur:\n{structure}\n"
            "Erkläre jede Variable und Methode im Code detailliert. Für jede Variable gib an:\n"
            "- Name\n- Typ (z. B. int, str)\n- Zweck\n- Verwendung\n"
            "Für jede Methode/Funktion gib an:\n"
            "- Name\n- Parameter\n- Rückgabewert\n- Zweck\n- Wie sie verwendet wird\n"
            "Formatiere die Antwort klar, z. B. als Liste oder Tabelle."
        )
        return self.call_llm(prompt, code)

    def technical_analysis(self, code: str, structure: str, explanations: str) -> str:
        """Führt eine technische Analyse des Codes durch."""
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

    def professional_analysis(self, code: str, structure: str, explanations: str, tech_analysis: str) -> str:
        """Führt eine professionelle Analyse des Codes durch."""
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

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator:
        """
        Hauptmethode der Pipeline. Analysiert den Code schrittweise und gibt die Ergebnisse als Generator zurück.
        
        Args:
            user_message (str): Der zu analysierende Code.
            model_id (str): Modell-ID (wird hier nicht verwendet, für Kompatibilität enthalten).
            messages (List[dict]): Konversationsverlauf (wird hier nicht verwendet).
            body (dict): Zusätzliche Parameter (wird hier nicht verwendet).
        
        Returns:
            Generator: Liefert die Analyseergebnisse schrittweise.
        """
        code = user_message

        def generate():
            # Schritt 1: Strukturanalyse
            structure = self.analyze_structure(code)
            if "Fehler" in structure:
                yield f"Strukturanalyse fehlgeschlagen: {structure}"
                return
            yield "**Code-Struktur:**\n" + structure + "\n\n"

            # Schritt 2: Erklärungen
            explanations = self.explain_elements(code, structure)
            if "Fehler" in explanations:
                yield f"Erklärung fehlgeschlagen: {explanations}"
                return
            yield "**Erklärungen zu Variablen/Methoden:**\n" + explanations + "\n\n"

            # Schritt 3: Technische Analyse
            tech_analysis = self.technical_analysis(code, structure, explanations)
            if "Fehler" in tech_analysis:
                yield f"Technische Analyse fehlgeschlagen: {tech_analysis}"
                return
            yield "**Technische Analyse:**\n" + tech_analysis + "\n\n"

            # Schritt 4: Professionelle Analyse
            prof_analysis = self.professional_analysis(code, structure, explanations, tech_analysis)
            if "Fehler" in prof_analysis:
                yield f"Professionelle Analyse fehlgeschlagen: {prof_analysis}"
                return
            yield "**Professionelle Analyse:**\n" + prof_analysis

        return generate()
