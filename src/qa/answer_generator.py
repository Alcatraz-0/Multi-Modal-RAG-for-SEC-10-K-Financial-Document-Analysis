"""
Answer generator using LLM to create responses from retrieved evidence
"""
from typing import List, Dict, Any, Optional


class AnswerGenerator:
    """Generate answers using LLM reader model"""

    def __init__(
        self,
        use_openai: bool = False,
        openai_api_key: Optional[str] = None,
        use_ollama: bool = False,
        ollama_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        local_model_path: Optional[str] = None,
        max_length: int = 512,
        temperature: float = 0.1
    ):
        """
        Initialize answer generator

        Args:
            use_openai: Whether to use OpenAI API
            openai_api_key: OpenAI API key
            use_ollama: Whether to use Ollama
            ollama_model: Ollama model name (e.g., llama3.2, mistral, phi3)
            ollama_base_url: Ollama API base URL
            local_model_path: Path to local model (transformers)
            max_length: Maximum answer length
            temperature: Sampling temperature
        """
        self.use_openai = use_openai
        self.use_ollama = use_ollama
        self.max_length = max_length
        self.temperature = temperature

        if use_openai:
            # Initialize OpenAI client
            import openai
            openai.api_key = openai_api_key
            self.client = openai
            print("✓ OpenAI client initialized")
        elif use_ollama:
            # Initialize Ollama client
            try:
                import requests
                self.ollama_model = ollama_model
                self.ollama_base_url = ollama_base_url
                self.requests = requests
                # Test connection
                response = requests.get(f"{ollama_base_url}/api/tags")
                if response.status_code == 200:
                    print(f"✓ Ollama client initialized (model: {ollama_model})")
                else:
                    print(f"⚠ Ollama server at {ollama_base_url} may not be running")
            except Exception as e:
                print(f"⚠ Error connecting to Ollama: {e}")
                print("Make sure Ollama is running: ollama serve")
        elif local_model_path:
            # Load local model (e.g., Llama, Mistral via transformers)
            self.model = None  # Placeholder
            print(f"Loading local model: {local_model_path}")
        else:
            raise ValueError("Must specify OpenAI, Ollama, or local model")

    def generate(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        route_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate answer from evidence

        Args:
            query: User query
            evidence: List of retrieved evidence pieces
            route_info: Query routing information

        Returns:
            Dictionary with answer and metadata
        """
        # Build prompt
        prompt = self._build_prompt(query, evidence, route_info)

        # Generate answer
        if self.use_openai:
            answer = self._generate_openai(prompt)
        elif self.use_ollama:
            answer = self._generate_ollama(prompt)
        else:
            answer = self._generate_local(prompt)

        return {
            'answer': answer,
            'confidence': self._estimate_confidence(answer, evidence)
        }

    def _build_prompt(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        route_info: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM"""

        # System instruction
        system_msg = """You are a financial analyst assistant. Answer questions about SEC 10-K filings based ONLY on the provided evidence.

IMPORTANT RULES:
1. Keep answers concise (2-5 sentences)
2. ALWAYS cite sources (section name, table ID, row/column)
3. Show units for all numbers (millions, thousands, etc.)
4. If evidence is insufficient, say so - do not guess
5. For numeric questions, show the calculation
"""

        # Format evidence
        evidence_text = "EVIDENCE:\n\n"
        for i, ev in enumerate(evidence[:5]):  # Top 5 pieces
            meta = ev['metadata']
            content = ev['content']
            content_type = meta['content_type']

            if content_type == 'table':
                evidence_text += f"[TABLE] {meta['ticker']} {meta['fiscal_year']} - Table {meta['table_id']}, Row {meta['row_idx']}\n"
            else:
                evidence_text += f"[TEXT] {meta['ticker']} {meta['fiscal_year']} - {meta.get('section_title', 'N/A')}\n"

            evidence_text += f"{content}\n\n"

        # Build full prompt
        prompt = f"""{system_msg}

{evidence_text}

QUESTION: {query}

ANSWER (include citations):"""

        return prompt

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_length,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama API"""
        try:
            response = self.requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_length
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Ollama error (status {response.status_code}): {response.text}"
        except Exception as e:
            return f"Error generating answer with Ollama: {str(e)}"

    def _generate_local(self, prompt: str) -> str:
        """
        Generate using local transformers model

        Note: Local model inference requires the transformers library
        and a loaded model. Use Ollama or OpenAI for production deployment.
        """
        if self.model is None:
            raise ValueError(
                "Local model not loaded. Please use use_ollama=True with Ollama, "
                "or use_openai=True with OpenAI API for answer generation."
            )

        # Use transformers pipeline for inference
        from transformers import pipeline
        generator = pipeline('text-generation', model=self.model)
        response = generator(prompt, max_length=self.max_length, temperature=self.temperature)
        return response[0]['generated_text']

    def _estimate_confidence(self, answer: str, evidence: List[Dict]) -> float:
        """Estimate confidence in answer"""
        # Simple heuristic: check if answer contains citations
        has_citation = any(marker in answer for marker in ['Table', 'Section', 'Row'])
        return 0.8 if has_citation else 0.5
