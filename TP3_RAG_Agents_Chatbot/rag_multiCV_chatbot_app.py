import os
import re
import time
from difflib import get_close_matches
from typing import Dict, List

import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq

from candidates import CANDIDATES


# Generacion de Embeddings
class EmbeddingsGenerator:
    """
    Clase para la generacion de Embeddings. Utiliza el modelo 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    """
    def __init__(self, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Inicializa el generador de Embeddings

        Args:
            modelo (str, optional): Nombre del modelo a cargar. Valor defaults es 
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".
        """
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


# Clase para la configuracion de Groq
class GroqLLM:
    """Wrapper para interactuar con el LLM de Groq."""
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Inicializa el cliente Groq.

        Args:
            api_key (str): API key de Groq.
            model (str): Modelo de LLM a utilizar.
        """
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, prompt: str) -> str:
        """
        Envía un prompt al LLM y devuelve la respuesta.

        Args:
            prompt (str): Texto del prompt.

        Returns:
            str: Respuesta generada por el LLM.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


# Funciones para Pinecone
def ensure_index(pc: Pinecone, name: str, dim: int, metric: str = "cosine"):
    """
    Asegura que el índice de Pinecone exista, lo crea si no existe.

    Args:
        pc (Pinecone): Cliente Pinecone.
        name (str): Nombre del índice.
        dim (int): Dimensión de los vectores.
        metric (str): Métrica de similitud.

    Returns:
        Índice de Pinecone listo para usar.
    """
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
    """
    Inserta los datos del CV de una persona en el índice de Pinecone.

    Args:
        index: Índice de Pinecone.
        embedder (EmbeddingsGenerator): Generador de embeddings.
        entries (List[Dict[str, str]]): Datos del CV.
        person (str): Nombre de la persona.
    """
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
    """
    Consulta el índice de Pinecone para obtener información relevante del CV.

    Args:
        pc: Cliente Pinecone.
        embedder (EmbeddingsGenerator): Generador de embeddings.
        person (str): Nombre de la persona.
        question (str): Pregunta del usuario.
        top_k (int): Número de resultados a devolver.

    Returns:
        Lista de resultados con metadatos.
    """
    index_name = f"cv-{person.lower()}"
    index = pc.Index(index_name)
    query_vec = embedder.generate_embedding(question)
    return index.query(vector=query_vec, top_k=top_k, include_metadata=True).get("matches", [])


def detect_person(query: str) -> str | None:
    """
    Detecta el nombre de un candidato en la consulta usando regex y coincidencias aproximadas.

    Args:
        query (str): Pregunta del usuario.

    Returns:
        str | None: Nombre detectado o None si no se encuentra.
    """
    for name in CANDIDATES.keys():
        if re.search(rf"\b{name}\b", query, re.IGNORECASE):
            return name
    words = query.split()
    for word in words:
        matches = get_close_matches(word, CANDIDATES.keys(), n=1, cutoff=0.8)
        if matches:
            return matches[0]
    return None


# Aplicación principal de Streamlit
def main():
    """
    Función principal que inicializa la aplicación Streamlit, los índices Pinecone,
    embeddings y LLM, y maneja la interfaz de chat con el usuario.
    """

    st.set_page_config(page_title="Multi-CV Agent Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot basado en RAG para Análisis de Múltiples CV's")

    # Inicializamos objetos en session_state
    if "pc" not in st.session_state:
        st.session_state.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if "embedder" not in st.session_state:
        st.session_state.embedder = EmbeddingsGenerator()
    if "llm" not in st.session_state:
        st.session_state.llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))

    # Creamos índices si no existen
    if "indexes_created" not in st.session_state:
        for person, entries in CANDIDATES.items():
            idx_name = f"cv-{person.lower()}"
            idx = ensure_index(st.session_state.pc, idx_name, st.session_state.embedder.dimension)
            populate_index(idx, st.session_state.embedder, entries, person)
        st.session_state.indexes_created = True

    # Inicializar historial de chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input del usuario
    query = st.text_input("Ask a question about a candidate's CV:")

    if query:
        # Limpiamos el historial para mostrar solo la respuesta actual
        st.session_state.chat_history = []
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Detectamos candidato
        person = detect_person(query)
        if not person:
            answer = "Could not detect a candidate name. Please include it."
        else:
            matches = query_person_index(st.session_state.pc, st.session_state.embedder, person, query)
            if not matches:
                answer = "No relevant information found in the CV."
            else:
                context_text = "\n".join([f"({m['metadata']['category']}) {m['metadata']['text']}" for m in matches])
                prompt = (
                    f"You are a helpful assistant answering questions about {person}'s CV.\n"
                    f"Use ONLY the following context to answer the question:\n{context_text}\n"
                    f"Question: {query}\n"
                    f"Provide a detailed, natural-language answer. "
                    f"If the answer is unknown, say 'I don't know.'"
                )
                answer = st.session_state.llm.chat(prompt)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Mostrar solo la última pregunta y respuesta
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])


if __name__ == "__main__":
    main()
