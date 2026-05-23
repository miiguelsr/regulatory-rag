"""
Hello RAG: el sistema RAG más simple posible, funcionando end-to-end.
Demuestra el ciclo completo: ingesta, embedding, indexación, retrieval, generación.
"""
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

documents = [
    "El IVA en México tiene una tasa general del 16% que aplica a la mayoría de bienes y servicios.",
    "La tasa de IVA fronteriza es del 8% y aplica en regiones fronterizas norte y sur de México.",
    "Los alimentos y medicinas tienen tasa 0% de IVA según la Ley del Impuesto al Valor Agregado.",
    "El IEPS es el Impuesto Especial sobre Producción y Servicios, distinto del IVA.",
    "Las personas morales en México deben presentar declaraciones mensuales del IVA.",
]

client = QdrantClient(url="http://localhost:6333")
collection_name = "hello_rag"

print("Generando embeddings con BGE-M3 vía Ollama...")
sample_embedding = ollama.embed(model="bge-m3", input=documents[0])
vector_size = len(sample_embedding["embeddings"][0])
print(f"Dimensión del vector: {vector_size}")

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

print("Indexando documentos...")
points = []
for idx, doc in enumerate(documents):
    embedding = ollama.embed(model="bge-m3", input=doc)["embeddings"][0]
    points.append(PointStruct(id=idx, vector=embedding, payload={"text": doc}))

client.upsert(collection_name=collection_name, points=points)
print(f"Indexados {len(documents)} documentos.\n")

query = "¿Cuál es la tasa de IVA en zonas fronterizas?"
print(f"Consulta: {query}\n")

query_embedding = ollama.embed(model="bge-m3", input=query)["embeddings"][0]

results = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=3,
).points

print("Chunks recuperados:")
for i, result in enumerate(results, 1):
    print(f"  {i}. (score={result.score:.3f}) {result.payload['text']}")
print()

context = "\n".join([f"- {r.payload['text']}" for r in results])
prompt = f"""Responde la siguiente pregunta usando exclusivamente la información del contexto.
Si la respuesta no está en el contexto, responde "No tengo información suficiente".

Contexto:
{context}

Pregunta: {query}

Respuesta:"""

print("Generando respuesta con Llama 3.2...")
response = ollama.generate(model="llama3.2:3b", prompt=prompt)
print(f"\nRespuesta:\n{response['response']}")
