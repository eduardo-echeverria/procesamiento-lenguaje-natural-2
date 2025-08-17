import os
import time
# import numpy as np
from groq import Groq
from typing import List, Dict, Any, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
# import openai

os.environ["TOKENIZERS_PARALLELISM"] = "false"
conversation_history = []


class EmbeddingsGenerator:
    """
    Clase para generar embeddings usando diferentes modelos.
    """
    
    def __init__(self, modelo: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el generador de embeddings.
        
        Args:
            modelo (str): Nombre del modelo de Sentence Transformers
        """
        self.modelo_nombre = modelo
        self.modelo = SentenceTransformer(modelo)
        self.dimension = self.modelo.get_sentence_embedding_dimension()
        
        print(f"✅ Modelo '{modelo}' cargado (dimensión: {self.dimension})")
    
    def generate_embedding(self, texto: str) -> List[float]:
        """
        Genera embedding para un texto individual.
        
        Args:
            texto (str): Texto a convertir en embedding
            
        Returns:
            List[float]: Vector de embedding
        """
        embedding = self.modelo.encode(texto)
        return embedding.tolist()
    
    def generate_embeddings_batch(self, textos: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos de manera eficiente.
        
        Args:
            textos (List[str]): Lista de textos
            
        Returns:
            List[List[float]]: Lista de vectores de embedding
        """
        embeddings = self.modelo.encode(textos)
        return [emb.tolist() for emb in embeddings]


# Configuracion Pinecone

def pinecone_config():
    """
    Configura la sesion en Pinecone. Se utilizan las siguientes variables de entorno:

    Env Vars:
    - PINECONE_API_KEY: La API Key obtenida al crear la cuenta en Pinecone
    - PINECONE_ENV: Entorno de Pinecone. Para este ejercicio utilizaremos us-east-1 de AWS
    """
    # Obtenemos las variables de entorno
    pc_api_key = os.getenv("PINECONE_API_KEY")

    # Validamos se tenga una API Key
    if not pc_api_key:
        raise ValueError("No se encuentra 'PINECONE_API_KEY' en las variables de entorno")
    
    # Creamos el objeto Pinecone client
    return Pinecone(api_key=pc_api_key)

def groq_client():
    # Obtenemos las variables de entorno
    gc_api_key = os.getenv("GROQ_API_KEY")

    # Validamos se tenga una API Key
    if not gc_api_key:
        raise ValueError("No se encuentra 'GROQ_API_KEY' en las variables de entorno")
    
    # Creamos el objeto Groq client
    return Groq(api_key=gc_api_key)

# Creacion del indice
def create_index(pc, index_name, dimension=384, metric="cosine"):
    
    timeout = 30  # seconds
    start_time = time.time()

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

    while True:
        existing_indexes = pc.list_indexes()
        if index_name in existing_indexes:
            break
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Index {index_name} not found after {timeout}s")
        time.sleep(1)

    print(f"Indice '{index_name}' creado exitosamente")

    return pc.Index(index_name)

# Construimos el contenido del indice
def build_index_content(index, index_name, embeddings_generator):
    docs = [
        {
            "id": "001",
            "text": "EDUARDO ECHEVERRIA\nElectronics Engineer\nBuenos Aires, Argentina | eduardo.a.echeverria@outlook.com | +549 11 6500 5672",
            "category": "User data",
            "date": "2025-08-17"
        },
        {
            "id": "002",
            "text": "Cisco Certified DevNet Associate, AWS Certified Solution Architect Associate,\
                Microsoft Certified Azure Fundamentals, CCNA Routing and Switching", 
            "category": "Certifications",
            "date": "2025-08-17"
        },
        {   
            "id": "003",
            "text": "Python, Flask, Django, FastAPI, Pydantics, JavaScript (Nodejs), Ansible",
            "category": "Automation Skills",
            "date": "2025-08-17"
        },
        {   
            "id": "004",
            "text": "BGP, OSPF, WAN, LAN, WLAN, DNS, DHCP, CLOS Architecture.",
            "category": "Networking Skills",
            "date": "2025-08-17"
        },
        {   
            "id": "005",
            "text": "AWS Architecture and application development, Azure administration and application development",
            "category": "Cloud Computing Skills",
            "date": "2025-08-17"
        },
        {   
            "id": "006",
            "text": "Kubernetes Developer, OpenShift, VMWare, Nutanix, OpenStack",
            "category": "Virtualization Skills",
            "date": "2025-08-17"
        },
        {
            "id": "007",
            "text": "Microsoft Windows 2016/2019 server, Linux servers",
            "category": "Compute skills",
            "date": "2025-08-17"
        },
        {
            "id": "007",
            "text": "English, Spanish, Portuguese",
            "category": "Language",
            "date": "2025-08-17"
        },
        {
            "id": "008",
            "text": "Network Developer Engineer, ExxonMobil Global Business Center\
                Developed a Python based application backend to automate the software package installation for \
                SDWAN Routers, Developed a Python based application backend to automates the management of configuration \
                templates for Versa SWAN Routers, Implemented the Vertical Pod Auto-scalation feature on OpenShift containers",
            "category": "Experience",
            "date": "2025-08-17"
        },
        {
            "id": "009",
            "text": "IP solutions Pre-Sales Engineer, Arris international, Deployed a Remote-PHY Trial environment, \
                Elaborated Point-by-Point technical answers for the CMTS and CIN (Converged Interconnect \
                Network) RFP for Telecentro Argentina",
            "category": "Experience",
            "date": "2025-08-17"
        },
        {
            "id": "010",
            "text": "Mobile Cloud Core Network Engineer, Huawei Technologies, \
                Installed and commissioned Data Centers supporting the laaS services, \
                Installed, commissioned and maintained DNS equipment",
            "category": "Experience",
            "date": "2025-08-17"
        },
        {
            "id": "011",
            "text": "Bachelor of Science degree in Electronics Engineering",
            "category": "Education",
            "date": "2025-08-17"
        },
        {
            "id": "012",
            "text": "VMCA VMWare Certified Associate Training",
            "category": "Education",
            "date": "2025-08-17"
        },
        {
            "id": "013",
            "text": "Introduction to Quantum Computing Course, Sponsored by IBM Quantum",
            "category": "Education",
            "date": "2025-08-17"
        }

    ]

    texts = [doc["text"] for doc in docs]

    if len(docs) > 1:
        embeddings = embeddings_generator.generate_embeddings_batch(texts)
    else:
        embeddings = [embeddings_generator.generate_embedding(texts[0])]

    vectors_to_add = []
    
    for i, doc in enumerate(docs):
        vector_data = {
            "id": doc["id"],
            "values": embeddings[i],
            "metadata": {
                "text": doc["text"],
                "category": doc["category"],
                "date": doc["date"],
                "length": len(doc["text"])
            }
        }
        vectors_to_add.append(vector_data)

    # Agregamos los vectores en el indice
    index.upsert(vectors=vectors_to_add)

    # Verificar estadísticas del índice
    stats = index.describe_index_stats()
    print(f"✅ Índice poblado exitosamente")
    print(f"   📊 Total de vectores: {stats['total_vector_count']}")
    print(f"   📏 Dimensión: {stats['dimension']}")
    
    return True

def chatbot(gc, query_input, index, embeddings_generator):
    # Variable for Groq
    global conversation_history

    query_embedding = embeddings_generator.generate_embedding(query_input)

    results = index.query(
        vector = query_embedding,
        top_k = 3,
        include_metadata = True
    )

    documents_found = []
    print(f"Resultados encontrados: {len(results['matches'])}")
    print("~" * 80)

    for i, match in enumerate(results["matches"]):
        document = {
            "posicion": i,
            "id": match["id"],
            "score": round(match["score"], 4),
            "texto": match["metadata"]["text"],
            "categoria": match["metadata"]["category"],
            "fecha": match["metadata"]["date"]
        }

        documents_found.append(document)

        # Mostrar resultado formateado
        print(f"{i}. ID: {document['id']}")
        print(f"   📊 Score: {document['score']}")
        print(f"   🏷️  Categoría: {document['categoria']}")
        print(f"   📅 Fecha: {document['fecha']}")
        print(f"   📝 Texto: {document['texto'][:100]}...")
        print("-" * 80)

    results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
    if not results["matches"]:
        return "NO RESULTS !!!!!!!!!"

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

def main():

    # Definimos el nombre del indice
    index_name = "cv-rag-chatbot"

    try:
        embeddings_generator = EmbeddingsGenerator()
        pc = pinecone_config()
        gc = groq_client()
        index = create_index(pc, index_name, dimension=embeddings_generator.dimension)
        build_index_content(index, index_name, embeddings_generator)
        # query_input = "Does the candidate have BGP and OSPF knowledge?"
        query_input = input(f"Formule la consulta:\n")
        final_answer = chatbot(gc, query_input, index, embeddings_generator)
        print (f"Final answer is: {final_answer}")
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        raise



if __name__ == "__main__":
    main()