import os
import re
import time
from typing import List, Dict

import streamlit as st
from difflib import get_close_matches

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import Any, List, Optional
from groq import Groq

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.llms.base import LLM
from candidates import CANDIDATES
from langchain.prompts import PromptTemplate
from langchain.schema import AgentFinish
from langchain.agents import AgentExecutor
from langchain.prompts import StringPromptTemplate


class SafeAgentExecutor(AgentExecutor):
    """
    Custom AgentExecutor for robust Streamlit integration.
    
    Features:
    - Always returns a dict with 'output' key to satisfy LangChain.
    - If LLM produces both Action and Final Answer, prefers the Final Answer.
    - Expands very short answers (like 'Yes'/'No') into richer explanations.
    - Falls back gracefully to any available text if no 'Final Answer' is detected.
    """

    def _return(self, output, intermediate_steps, run_manager=None):
        text = None

        # Case 1: LLM returned AgentFinish
        if isinstance(output, AgentFinish):
            text = output.return_values.get("output", "")

        # Case 2: LLM returned a dict with 'output'
        elif isinstance(output, dict) and "output" in output:
            text = output["output"]

        # Extract Final Answer if present
        if text:
            if "Final Answer:" in text:
                # Take everything after the last "Final Answer:" marker
                text = text.split("Final Answer:")[-1].strip()

                # Expand short answers into richer explanations
                if len(text.split()) <= 3:
                    text = (
                        f"The candidate information suggests: {text}. "
                        "Please note this answer is based solely on the CV data."
                    )

            # Always return as a dict
            return {"output": text.strip()}

        # Fallback: stringify any other output and wrap in 'output'
        result = super()._return(output, intermediate_steps, run_manager)
        return {"output": str(result)}


# Groq LLM Wrapper para LangChain
class GroqLLM(LLM):
    """Wrap Groq client into a LangChain-compatible LLM."""

    api_key: str
    model: str = "llama-3.3-70b-versatile"
    client: Optional[Groq] = None  # Declare client as an optional field

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", **kwargs):
        super().__init__(api_key=api_key, model=model, **kwargs)
        object.__setattr__(self, "client", Groq(api_key=api_key))

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# Generacion de Embeddings
class EmbeddingsGenerator:
    """
    Clase para la generacion de Embeddings. Utiliza el modelo 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    por ser un modelo que soporta Ingles y Español.
    """
    def __init__(self, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera un embedding para una entrada de textp

        Args:
            texto (str): El texto de entrada a codificar

        Returns:
            List[float]: Vector de embedding representando el texto de entrada.
        """
        return self.model.encode(text).tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para entradas de texto en batches.

        Args:
            textos (List[str]): Lista de strings a codificar

        Returns:
            List[List[float]]: Lista de vectores de embedding vectors, uno por cada text introducido.
        """
        return [emb.tolist() for emb in self.model.encode(texts)]


def ensure_index(pc: Pinecone, name: str, dim: int, metric: str = "cosine"):
    existing = [i["name"] for i in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while name not in [i["name"] for i in pc.list_indexes()]:
            time.sleep(1)
    return pc.Index(name)


def populate_index(index, embedder: EmbeddingsGenerator, entries: List[Dict[str, str]], person: str):
    texts = [f"{e['category']}: {e['text']}" for e in entries]
    embs = embedder.generate_embeddings_batch(texts)
    vectors = []
    for i, e in enumerate(entries):
        vectors.append({
            "id": f"{person}-{e['category']}-{i}",
            "values": embs[i],
            "metadata": {"category": e["category"], "text": e["text"]}
        })
    index.upsert(vectors=vectors)


def query_person_index(pc, embedder, person: str, question: str, top_k: int = 3):
    index_name = f"cv-{person.lower()}"
    index = pc.Index(index_name)
    query_vec = embedder.generate_embedding(question)
    return index.query(vector=query_vec, top_k=top_k, include_metadata=True).get("matches", [])


def detect_person(query: str) -> str | None:
    """Detect candidate name in query using regex + fuzzy match"""
    for name in CANDIDATES.keys():
        if re.search(rf"\b{name}\b", query, re.IGNORECASE):
            return name
    words = query.split()
    for word in words:
        matches = get_close_matches(word, CANDIDATES.keys(), n=1, cutoff=0.8)
        if matches:
            return matches[0]
    return None


def main():
    # API Keys
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    gc_llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))

    # Instanciamos el generador de embeddings
    embedder = EmbeddingsGenerator()

    # Creamos los Pinecone index
    for person, entries in CANDIDATES.items():
        idx_name = f"cv-{person.lower()}"
        idx = ensure_index(pc, idx_name, embedder.dimension)
        populate_index(idx, embedder, entries, person)

    # Definimos los datos para el agente
    def pinecone_tool_func(query: str) -> str:
        person = detect_person(query)
        if not person:
            return "Could not detect a candidate name. Please include it."
        matches = query_person_index(pc, embedder, person, query)
        if not matches:
            return "No results found in the CV."
        return "\n".join([f"({m['metadata']['category']}) {m['metadata']['text']}" for m in matches])

    tools = [
        Tool(
            name="CVSearchTool",
            func=pinecone_tool_func,
            description="Searches a candidate's CV in Pinecone given a query about their skills, experience, or certifications."
        )
    ]

    tool_names = ", ".join([t.name for t in tools])
    react_prompt = PromptTemplate.from_template("""
    You are an assistant that answers questions about people's CVs using tools.

    You must STRICTLY follow this format:

    Question: the input question
    Thought: your reasoning
    Action: the action to take, must be one of [{tool_names}]
    Action Input: the input for that action
    Observation: result of the action
    ... (repeat Thought/Action/Observation as needed)
    Thought: I now know the final answer
    Final Answer: Provide a detailed and natural-language answer to the question, 
    grounded ONLY in the knowledge base. 
    If the information is not present, say "I don’t know."

    When you are ready to provide the final response, you MUST finish with:

    Final Answer: <your detailed, natural-language answer here>

    Rules:
    - Always include "Final Answer:" exactly once at the end.
    - The final answer should be a full explanation, not just "Yes" or "No".
    - Base your response ONLY on the context and tool results.
    - If the information is not present, say "I don’t know."

    """.format(tool_names=tool_names))

    # Initialize LangChain agent
    agent = initialize_agent(
        tools=tools,
        llm=gc_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": react_prompt.template},
        tool_names = ", ".join([t.name for t in tools])
    )

    agent = SafeAgentExecutor.from_agent_and_tools(agent=agent.agent, tools=tools, verbose=True)

    query = input("Ask a question about one candidate\n>>> ")
    if query:
        answer = agent.run(query)
        print("="*80)
        print("Answer: \n")
        print(f'"{answer}"\n')
        print("="*80)

if __name__ == "__main__":
    main()
