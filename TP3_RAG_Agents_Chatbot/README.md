# 📄 Chatbot basado en RAG para Análisis de Múltiples CV's

Este proyecto es un chatbot interactivo que responde preguntas sobre los CVs de varios candidatos.  
La aplicación utiliza:

- **Pinecone**: Base de datos vectorial para almacenar embeddings del CV.  
- **Sentence-Transformers**: Para generar embeddings semánticos multilingües.  
- **Groq LLM**: Modelo de lenguaje para generar respuestas naturales y detalladas.  
- **Streamlit**: Interfaz web interactiva para el chatbot.  

---

## 🚀  Características

- Indexa la información de varios CV en fragmentos de texto (*text chunks*) dentro de **Pinecone**.  
- Genera embeddings semánticos usando `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.  
- Responde preguntas en lenguaje natural sobre un candidato (ej: *"Does Eduardo know C#?"*).  
- Maneja múltiples candidatos y consultas secuenciales de forma robusta.  
- La lógica del chatbot utiliza `SafeAgentExecutor` para evitar crashes por parsing de LangChain.  

---

## 📦 Instalación

1. **Clonar el repositorio**:

```
bash
git clone https://github.com/eduardo-echeverria/procesamiento-lenguaje-natural-2.git
cd TP3_RAG_Agents_Chatbot
```

2. **Crear y activar un entorno virtual (recomendado)**:

```
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. **Instalar dependencias**:

```
pip install -r requirements.txt
```

## 🔑 Variables de Entorno

La aplicación requiere las API Keys de Pinecone y Groq:

```
export PINECONE_API_KEY="tu_pinecone_api_key"
export GROQ_API_KEY="tu_groq_api_key"
```

⚠️ No hardcodear las API Keys. Se recomienda usar un archivo `.env` o variables de entorno para mantenerlas seguras.

## ▶️ Ejecutar la aplicación

Para iniciar el servidor de Streamlit:
```
streamlit run rag_multiCV_chatbot_app.py
```
Streamlit abrirá una ventana del navegador en:
```
http://localhost:8501
```
## 📂 Estructura del proyecto
```
.
├── rag_multiCV_chatbot_app.py      # Aplicación Streamlit principal con SafeAgentExecutor
├── candidates.py                   # Diccionario con los candidatos y sus CVs
├── rag_multiCV_chatbot_test.mp4    # Video demostrativo de la aplicación
├── rag_agents_app.py               # Codigo con la implementacion de LangChain agents
├── README.md                       # Documentación del proyecto
└── requirements.txt                # Dependencias de Python
```

## ⚠️ Notas técnicas

Se usa SafeAgentExecutor para manejar correctamente los outputs del LLM y evitar errores de parsing.

Los embeddings son multilingües y permiten consultas en inglés o español.

La detección de candidatos utiliza regex y difflib.get_close_matches para coincidencias aproximadas.

Si el modelo Groq llama-3.3-70b-versatile es muy lento, se puede usar uno más pequeño (ej: llama-3-8b o llama-3-13b).

El archivo "rag_agents_app.py" contine codigo en el cual se implementan agnetes LangChain. Al momento de utilizar LangChain en combinacion con Streamlit tuve muchos problemas por la forma que LangChain devuelve las respuestas en forma de objetos de tipo: Action + Thought + Final Answer. Sumado a eso, Streamlit en su funcionamiento por defecto siempre ejecuta todo el script para cata interacccion, lo cual causa que el agente vea respuestas anteriores o estados parciales lo cual origina un error de parseo de forma constante. El codigo en "rag_agents_app.py" finalmente prescinde de Streamlit, sin embargo puede ser ejecutado como un archivo Python. En la ejecucion se puede apreciar que la palicacion responde de igual manera las preguntas realizadas sobre los CVs.