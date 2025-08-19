import os
import time
from groq import Groq
from typing import List, Dict, Any, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import streamlit as st

# Definimos algunos parametros globales
os.environ["TOKENIZERS_PARALLELISM"] = "false"
conversation_history = []

# Generacion de Embeddings
class EmbeddingsGenerator:
    """
    Clase para la generacion de Embeddings. Utiliza el modelo 'SentenceTransformer'
    de Hugging Face.
    """
    def __init__(self, modelo: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el generador de Embeddings

        Args:
            modelo (str, optional): Nombre del modelo a cargar. Valor defaults es 
                `"sentence-transformers/all-MiniLM-L6-v2"`.
        """
        self.modelo = SentenceTransformer(modelo)
        self.dimension = self.modelo.get_sentence_embedding_dimension()

        print(f"✅ Modelo '{modelo}' cargado (dimensión: {self.dimension})")
    
    def generate_embedding(self, texto: str) -> List[float]:
        """
        Genera un embedding para una entrada de textp

        Args:
            texto (str): El texto de entrada a codificar

        Returns:
            List[float]: Vector de embedding representando el texto de entrada.
        """
        return self.modelo.encode(texto).tolist()
    
    def generate_embeddings_batch(self, textos: List[str]) -> List[List[float]]:
        """
        Genera embeddings para entradas de texto en batches.

        Args:
            textos (List[str]): Lista de strings a codificar

        Returns:
            List[List[float]]: Lista de vectores de embedding vectors, uno por cada text introducido.
        """
        return [emb.tolist() for emb in self.modelo.encode(textos)]
    
# Configuracion Pinecone
def pinecone_config():
    """
    Configura e inicializa el cliente de Pinecone utilizando variables de entorno.

    Variables de entorno:
        PINECONE_API_KEY (str): Pinecone API Key.

    Raises:
        ValueError: Si la variable de entorno 'PINECONE_API_KEY' no esta setteada.

    Returns:
        Pinecone: una instancia inicializada de Pinecone.
    """
    # Obtenemos las variables de entorno
    pc_api_key = os.getenv("PINECONE_API_KEY")

    # Validamos se tenga una API Key
    if not pc_api_key:
        raise ValueError("No se encuentra 'PINECONE_API_KEY' en las variables de entorno")
    
    # Creamos el objeto Pinecone client
    return Pinecone(api_key=pc_api_key)

# Configuracion Groq
def groq_client():
    """
    Configura e inicializa el cliente de Groq utilizando variables de entorno.

    Variables de entorno:
        GROQ_API_KEY (str): Groq API Key.

    Raises:
        ValueError: Si la variable de entorno 'GROQ_API_KEY' no esta setteada.

    Returns:
        Groq: una instancia inicializada de Groq.
    """
    # Obtenemos las variables de entorno
    gc_api_key = os.getenv("GROQ_API_KEY")

    # Validamos se tenga una API Key
    if not gc_api_key:
        raise ValueError("No se encuentra 'GROQ_API_KEY' en las variables de entorno")
    
    # Creamos el objeto Groq client
    return Groq(api_key=gc_api_key)

# Creacion del indice
def create_index(pc, index_name, dimension=384, metric="cosine"):
    """
    Crea y/o obtiene el Pinecone index.

    Esta funcion verifica si un index con el nombre proporcionado existe.
    Si existiese, este index es retornado. En caso contrario crea un nuevo
    index con la configuracion especificada.

    Args:
        pc (Pinecone): Instancia inicializada del cliente Pinecone.
        index_name (str): nombre del index a crear o recuperar.
        dimension (int, optional): Dimension de los embeddings. Valor defaults es 384.
        metric (str, optional): Metrica utilizada para la busqueda vectorial. Valor defaults es "cosine".

    Raises:
        TimeoutError: Si el indez no es encontrado en 30 segundos.

    Returns:
        pinecone.Index: Un objeto de index Pinecone (Existente or nuevo).
    """
    # Verificamos si el indice existe
    existing_indexes = pc.list_indexes()
    for idx in existing_indexes:
        if index_name == idx["name"]:
            print(f"El indice '{index_name}' ya existe")
            return pc.Index(index_name)
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    timeout = 30
    start_time = time.time()
    while True:
        existing_indexes = pc.list_indexes()
        if index_name in existing_indexes:
            break
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Index {index_name} not found after {timeout}s")
        time.sleep(1)

    return pc.Index(index_name)

# Construimos el contenido del indice
def build_index_content(index, index_name, embeddings_generator):
    """
    Completa el Pinecone Index con un documento de ejemplo y sus embeddings

    Args:
        index (pinecone.Index): Objeto Pinecone index donde todos los vectores seran insertados.
        index_name (str): Nombre del index.
        embeddings_generator (EmbeddingsGenerator): instancia 'EmbeddingsGenerator'.

    Returns:
        bool: True si los vectores fueron exitosamente agregados al index.
    """
    docs = [
        {
            "id": "001",
            "text": "EDUARDO ECHEVERRIA\nElectronics Engineer\nBuenos Aires, Argentina | eduardo.a.echeverria@outlook.com | +549 11 6500 5672",
            "category": "User data"
        },
        {
            "id": "002",
            "text": "Cisco Certified DevNet Associate, AWS Certified Solution Architect Associate,\
                Microsoft Certified Azure Fundamentals, CCNA Routing and Switching", 
            "category": "Certifications"
        },
        {   
            "id": "003",
            "text": "Python, Flask, Django, FastAPI, Pydantics, JavaScript (Nodejs), Ansible",
            "category": "Automation Skills"
        },
        {   
            "id": "004",
            "text": "BGP, OSPF, WAN, LAN, WLAN, DNS, DHCP, CLOS Architecture.",
            "category": "Networking Skills"
        },
        {   
            "id": "005",
            "text": "AWS Architecture and application development, Azure administration and application development",
            "category": "Cloud Computing Skills"
        },
        {   
            "id": "006",
            "text": "Kubernetes Developer, OpenShift, VMWare, Nutanix, OpenStack",
            "category": "Virtualization Skills"
        },
        {
            "id": "007",
            "text": "Microsoft Windows 2016/2019 server, Linux servers",
            "category": "Compute skills"
        },
        {
            "id": "008",
            "text": "English, Spanish, Portuguese",
            "category": "Language"
        },
        {
            "id": "009",
            "text": "Network Developer Engineer, ExxonMobil Global Business Center\
                Developed a Python based application backend to automate the software package installation for \
                SDWAN Routers, Developed a Python based application backend to automates the management of configuration \
                templates for Versa SWAN Routers, Implemented the Vertical Pod Auto-scalation feature on OpenShift containers",
            "category": "Experience"
        },
        {
            "id": "010",
            "text": "IP solutions Pre-Sales Engineer, Arris international, Deployed a Remote-PHY Trial environment, \
                Elaborated Point-by-Point technical answers for the CMTS and CIN (Converged Interconnect \
                Network) RFP for Telecentro Argentina",
            "category": "Experience"
        },
        {
            "id": "011",
            "text": "Mobile Cloud Core Network Engineer, Huawei Technologies, \
                Installed and commissioned Data Centers supporting the laaS services, \
                Installed, commissioned and maintained DNS equipment",
            "category": "Experience"
        },
        {
            "id": "012",
            "text": "Bachelor of Science degree in Electronics Engineering",
            "category": "Education"
        },
        {
            "id": "013",
            "text": "VMCA VMWare Certified Associate Training",
            "category": "Education"
        },
        {
            "id": "014",
            "text": "Introduction to Quantum Computing Course, Sponsored by IBM Quantum",
            "category": "Education"
        }
    ]

    texts = [doc["text"] for doc in docs]
    embeddings = embeddings_generator.generate_embeddings_batch(texts)

    vectors_to_add = []
    
    for i, doc in enumerate(docs):
        vector_data = {
            "id": doc["id"],
            "values": embeddings[i],
            "metadata": {
                "text": doc["text"],
                "category": doc["category"]
            }
        }
        vectors_to_add.append(vector_data)

    # Agregamos los vectores en el indice
    index.upsert(vectors=vectors_to_add)
    
    return True

def chatbot(gc, query_input, index, embeddings_generator):
    """
    Este Chatbot responde a la consulta de un usuario utilizando busqueda por 
    base de datos vectorial desde un index de Pinecone y utiliza el modelo de
    lenguaje Groq para generar respuestas.

    Args:
        gc (Groq): Instancia de cliente Groq utilizada para interactuar con el modelo de lenguaje.
        query_input (str): Consulta del usuario.
        index (pinecone.Index): Index Pinecone utilizado para busqueda vectorial.
        embeddings_generator (EmbeddingsGenerator): Instancia de 'EmbeddingsGenerator'.

    Returns:
        str: String con la respuesta generadoa por el modelo. Si no se encuentra un documento
        relevante, retirna un mensaje default inidicando data insuficiente.
    """
    # Variable for Groq
    global conversation_history

    query_embedding = embeddings_generator.generate_embedding(query_input)

    results = index.query(
        vector = query_embedding,
        top_k = 1,
        include_metadata = True
    )

    if not results["matches"]:
        return "I don't have enough data to answer your question."

    top_doc = results["matches"][0]["metadata"]["text"]
    # Variable for Groq
    augmented_query = f"""You are a helpful assistant. 

    Task: Use ONLY the information inside <knowledge_base> tags to answer. 
    If the information is not present, say "I don’t know."

    <knowledge_base>
    {top_doc}
    </knowledge_base>

    Question: {query_input}

    Answer:
    """
    
    conversation_history.append({"role": "user", "content": augmented_query})

    response = gc.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_history
    )
    
    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})

    return answer

# Streamlit UI
def main():
    # Definimos el nombre del indice
    index_name = "cv-rag-chatbot"

    st.title("📄 CV Chatbot")
    st.write("Ask questions about the candidate’s CV.")

    # Init session state
    if "initialized" not in st.session_state:
        embeddings_generator = EmbeddingsGenerator()
        pc = pinecone_config()
        gc = groq_client()
        index = create_index(pc, index_name, dimension=embeddings_generator.dimension)
        build_index_content(index, index_name, embeddings_generator)

        st.session_state.embeddings = embeddings_generator
        st.session_state.pc = pc
        st.session_state.gc = gc
        st.session_state.index = index
        st.session_state.initialized = True
    
    query = st.text_input("Enter your question:")

    if st.button("Ask"):
        if query.strip():
            answer = chatbot(st.session_state.gc, query, st.session_state.index, st.session_state.embeddings)
            st.write("### 🤖 Answer:")
            st.success(answer)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()


    
