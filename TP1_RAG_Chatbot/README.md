# 📄 CV Chatbot with Pinecone, Groq & Streamlit

Este proyecto es un chatbot interactivo que responde preguntas acerca el CV de un candidato.
La aplicacion utiliza **Pinecone** como Base de Datso Vectoriales, **SentenceTransformers** 
para embeddings, **Groq** como modelo de lenguaje y **Streamlit** para el diseño de la web UI.

---

## 🚀 Caracteristicas
- Indexa la informacion de un CV en pedazos de texto (text chunks) in **Pinecone**.
- Genera **semantic embeddings** utilizando `sentence-transformers/all-MiniLM-L6-v2`.
- Consulta el CV empleando preguntas formuladas en lenguaje natural (e.g., *"Does the candidate know Linux?"*).
- Para el ejemplo se utilizo una parte de mi CV, el cual tengo redactado en Ingles. Por este motivo las consultas son realizadas en Ingles.
- Devuelve la seccion mas relvante del CV y emplea al modelo **Groq’s LLaMA** para generar respuestas.
- Web frontend desarrollado en **Streamlit**.

---

## 📦 Instalacion

1. **Clonar este repositorio**:

```
bash
git clone https://github.com/eduardo-echeverria/procesamiento-lenguaje-natural-2.git
cd TP1_RAG_Chatbot
```

2. **Crear un entorno virtual de Python (Recomendado)**:

```
python3 -m venv venv
source venv/bin/activate   # on Mac/Linux
venv\Scripts\activate      # on Windows
```

3. **Instalar dependencias**:

```
pip install -r requirements.txt
```

## 🔑 Variables de Entorno

Esta aplicacion requiere API Keys de Pinecone y Groq:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```
⚠️ Se recomienda no hardcodear las API Keys. Utilice `.env` o variables de entorno
para matener las Keys seguras.

## ▶️ Ejecutar la App

Inicialice el servidor de Streamlit:
```
streamlit run app.py
```
Streamlit abrira una ventana de browser en:
```
http://localhost:8501
```

## 📂 Estructura del proyecto
```
.
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .env                # API keys (not committed)
```

## ⚠️ Notas

Si el modelo Groq’s 70B (llama-3.3-70b-versatile) se vuelve muy lento, se recomiend probar
uno mas pequeño (e.g. llama-3-8b or llama-3-13b).
