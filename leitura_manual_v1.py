# RAG com modelo alterável facilmente
# Troque abaixo para o modelo desejado em cada experimento

MODELO_SEM_CONTEXTO = "mistral:latest"     # Ou "mistral:instruct", "llama3", etc.
MODELO_COM_CONTEXTO = "mistral:latest"     # Pode ser diferente se quiser comparar

API_PORTA = 11434  # Porta padrão do Ollama

import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import requests

# 1. Ler manual PDF
pdf_filename = "prototipo_rag_manual\\manuais\\exemplo_manual_impressora_3d.pdf"  # Altere se necessário
def extrair_texto_pdf(filename):
    texto = ""
    with open(filename, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            texto += page.extract_text() + "\n"
    return texto

print("Lendo o manual PDF...")
texto_manual = extrair_texto_pdf(pdf_filename)
print("PDF lido.")

# 2. Dividir em chunks
def dividir_em_chunks(texto, tamanho_chunk=400, sobreposicao=100):
    palavras = texto.split()
    chunks = []
    i = 0
    while i < len(palavras):
        chunk = " ".join(palavras[i:i+tamanho_chunk])
        chunks.append(chunk)
        i += tamanho_chunk - sobreposicao
    return [c for c in chunks if len(c.strip()) > 0]

chunks = dividir_em_chunks(texto_manual)
print(f"{len(chunks)} chunks gerados.")

# 3. Embeddings e indexação
print("Gerando embeddings e indexando...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedder.encode(chunks, show_progress_bar=True)
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
index.add(np.array(chunk_embeddings))
print("Indexação FAISS concluída.")

# 4. Função para buscar chunks mais relevantes
def buscar_trechos(pergunta, k=2):
    pergunta_emb = embedder.encode([pergunta])
    D, I = index.search(np.array(pergunta_emb), k)
    return [chunks[i] for i in I[0]]

# 5. Função para montar o prompt (contexto RAG)
def montar_prompt(pergunta, trechos):
    prompt = f"""
Você é um assistente sobre o seguinte manual técnico.
Responda à pergunta usando apenas informações dos trechos abaixo do manual:

Manual:
{chr(10).join(trechos)}

Pergunta: {pergunta}

Responda em português, de forma clara, direta e fiel ao manual.
"""
    return prompt

# 6. Função genérica para perguntar ao modelo escolhido via API
def perguntar_ollama(prompt, modelo, porta=11434):
    url = f"http://localhost:{porta}/api/generate"
    response = requests.post(
        url,
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False,
            "max_tokens": 512,
        }
    )
    return response.json().get('response', "[ERRO ao obter resposta do modelo]")

# 7. Pergunta SEM contexto RAG
def perguntar_sem_contexto(pergunta):
    prompt = f"Pergunta: {pergunta}\nResponda em português de forma clara e direta."
    return perguntar_ollama(prompt, MODELO_SEM_CONTEXTO, API_PORTA)

# 8. Pergunta COM contexto RAG
def perguntar_com_contexto(pergunta):
    trechos = buscar_trechos(pergunta, k=2)
    prompt = montar_prompt(pergunta, trechos)
    return perguntar_ollama(prompt, MODELO_COM_CONTEXTO, API_PORTA)

# 9. Loop principal: compara as duas respostas para a mesma pergunta
if __name__ == "__main__":
    print("\n=== Sistema RAG com modelo selecionável ===")
    print(f"Modelo SEM contexto: {MODELO_SEM_CONTEXTO}")
    print(f"Modelo COM contexto: {MODELO_COM_CONTEXTO}")
    print("Faça perguntas sobre o manual PDF carregado.")
    print("Digite 'sair' para encerrar.\n")
    while True:
        pergunta_usuario = input("\nDigite sua pergunta sobre o manual (ou 'sair' para encerrar): ")
        if pergunta_usuario.lower() == "sair":
            print("Encerrando o sistema. Até a próxima!")
            break

        print(f"\n--- RESPOSTA {MODELO_SEM_CONTEXTO.upper()} *SEM CONTEXTO* (chatbot puro) ---")
        resposta_sem_contexto = perguntar_sem_contexto(pergunta_usuario)
        print(resposta_sem_contexto)

        print(f"\n--- RESPOSTA {MODELO_COM_CONTEXTO.upper()} *COM CONTEXTO DO MANUAL* (RAG) ---")
        resposta_com_contexto = perguntar_com_contexto(pergunta_usuario)
        print(resposta_com_contexto)
