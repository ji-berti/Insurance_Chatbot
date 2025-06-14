import os
import pandas as pd
import numpy as np
import warnings
import time
from datasets import Dataset

# Langchain and RAGAS imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

# Importar configuraciones del proyecto
import sys
sys.path.append('..')
from generative_resp import config_model, config_vectordb

# Suprimir warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def configure_ragas_dependencies():
    """
    Configura dependencias de RAGAS con parámetros conservadores.
    """
    print("Configurando dependencias de RAGAS (LLM y Embeddings)...")
    try:
        # 1. Configurar el LLM de Gemini para máxima consistencia
        gemini_llm = ChatGoogleGenerativeAI(
            model=config_model.GEMINI_MODEL,
            google_api_key=config_model.GEMINI_API_KEY,
            temperature=0.0,  # Sin creatividad para respuestas predecibles
            # FIX: El parámetro correcto es 'timeout', no 'request_timeout'
            timeout=300,
        )
        ragas_llm = LangchainLLMWrapper(gemini_llm)
        
        # 2. Configurar los Embeddings de HuggingFace
        hf_embeddings = HuggingFaceEmbeddings(
            model_name=config_vectordb.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        
        print("✅ Dependencias de RAGAS configuradas exitosamente.")
        return ragas_llm, ragas_embeddings
        
    except Exception as e:
        print(f"❌ Error configurando dependencias de RAGAS: {e}")
        return None, None

def create_rag_chain(docs, temperature, top_k, chunk_size):
    """
    Crea una cadena RAG con los hiperparámetros especificados.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=config_vectordb.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=config_vectordb.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': top_k})
    
    llm = ChatGoogleGenerativeAI(
        model=config_model.GEMINI_MODEL,
        google_api_key=config_model.GEMINI_API_KEY,
        temperature=temperature,
        max_output_tokens=config_model.MAX_OUT_TOKENS,
    )
    
    prompt_template = """Usa los siguientes fragmentos de contexto para responder la pregunta. Si no sabes la respuesta, simplemente di que no lo sabes. Responde en español.

Contexto: {context}

Pregunta: {question}

Respuesta útil:"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def get_manual_hyperparameter_configs():
    """
    Devuelve una lista de 4 combinaciones de hiperparámetros definidas manualmente.
    """
    return [
        {'name': 'Baseline', 'chunk_size': 1000, 'top_k': 4, 'temperature': 0.1},
        {'name': 'More Context, Focused', 'chunk_size': 1500, 'top_k': 3, 'temperature': 0.1},
        {'name': 'Smaller Chunks, More Options', 'chunk_size': 500, 'top_k': 5, 'temperature': 0.1},
        {'name': 'Creative & Balanced', 'chunk_size': 1000, 'top_k': 4, 'temperature': 0.3},
    ]

def create_evaluation_dataset():
    """
    Crea un DataFrame de pandas con 4 preguntas y respuestas para la evaluación.
    """
    data = {
        'question': [
            "¿Cuáles son las condiciones que deben darse para que la compañía aseguradora reembolse los gastos médicos?",
            "¿Cuáles son las coberturas que otorga la compañía aseguradora para prestaciones médicas de alto costo?",
            "¿Cuáles son las exclusiones del seguro de COVID-19?",
            "¿Cuándo debe ser denunciado el siniestro de enfermedades graves?",
        ],
        'ground_truth': [
            "Que haya transcurrido el periodo de carencia, que la póliza esté vigente y que no haya transcurrido el plazo para la cobertura del Evento.",
            "Beneficio de hospitalización (días cama, servicios, honorarios médicos), prótesis, cirugía dental por accidente, servicio de enfermera y ambulancia, y beneficio ambulatorio.",
            "Gastos de hospitalización, rehabilitación o fallecimiento asociados a enfermedades distintas al COVID-19.",
            "El asegurado debe notificar a la compañía tan pronto sea posible una vez tomado conocimiento del diagnóstico de la enfermedad grave cubierta.",
        ]
    }
    return pd.DataFrame(data)

def run_evaluation(original_docs):
    """
    Función principal que ejecuta la evaluación con la lógica más robusta posible.
    """
    ragas_llm, ragas_embeddings = configure_ragas_dependencies()
    if not ragas_llm or not ragas_embeddings:
        return None

    ragas_run_config = RunConfig(max_workers=1)
    configs = get_manual_hyperparameter_configs()
    eval_dataset_pd = create_evaluation_dataset()
    
    all_results = []
    
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    metric_names = [m.name for m in metrics]
    
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = ragas_embeddings

    num_configs = len(configs)
    for i, config in enumerate(configs):
        print("\n" + "="*50)
        print(f"🧪 Evaluando Configuración #{i+1}/{num_configs}: {config['name']}")
        print(f"   Hiperparámetros: chunk_size={config['chunk_size']}, top_k={config['top_k']}, temp={config['temperature']}")
        print("="*50)
        
        rag_chain, retriever = create_rag_chain(
            docs=original_docs,
            temperature=config['temperature'],
            top_k=config['top_k'],
            chunk_size=config['chunk_size']
        )
        
        combination_scores = []
        for index, row in eval_dataset_pd.iterrows():
            print(f"  - Evaluando pregunta {index + 1}/{len(eval_dataset_pd)}...")
            scores_dict = {name: np.nan for name in metric_names} # Iniciar con NaN
            
            try:
                response = rag_chain.invoke(row['question'])
                retrieved_docs = retriever.invoke(row['question'])
                contexts = [doc.page_content for doc in retrieved_docs]

                single_question_dataset = Dataset.from_dict({
                    'question': [row['question']],
                    'answer': [response],
                    'contexts': [contexts],
                    'ground_truth': [row['ground_truth']]
                })

                result = evaluate(
                    dataset=single_question_dataset, 
                    metrics=metrics, 
                    raise_exceptions=True,
                    run_config=ragas_run_config
                )
                
                # --- LÓGICA DE PROCESAMIENTO ULTRA-ROBUSTA ---
                if hasattr(result, 'scores'):
                    scores_obj = result.scores
                    # CASO CLAVE: Si RAGAS devuelve una lista de scores
                    if isinstance(scores_obj, list) and len(scores_obj) > 0:
                        scores_dict = scores_obj[0]
                        print("    ✅ Éxito (resultado procesado desde lista).")
                    # CASO NORMAL: Si devuelve un objeto con .to_dict()
                    elif hasattr(scores_obj, 'to_dict'):
                        scores_dict = scores_obj.to_dict()
                        print("    ✅ Éxito (resultado procesado desde objeto).")
                    # CASO ALTERNATIVO: Si es directamente un diccionario
                    elif isinstance(scores_obj, dict):
                        scores_dict = scores_obj
                        print("    ✅ Éxito (resultado procesado desde dict).")
                    else:
                        print(f"    ⚠️ Fallo (formato de 'scores' inesperado: {type(scores_obj)}).")
                else:
                    print(f"    ⚠️ Fallo (objeto de resultado no tiene atributo 'scores').")

            except Exception as e:
                print(f"    ⚠️ Fallo por excepción: {str(e)[:100]}...")
            
            finally:
                # Asegurarse de que el diccionario final tenga todas las métricas
                final_scores = {name: scores_dict.get(name, np.nan) for name in metric_names}
                combination_scores.append(final_scores)
                print(f"    ⏳ Pausa de 180 segundos para proteger la cuota...")
                time.sleep(180) # Pausa muy larga para máxima seguridad

        # Calcular promedio para la configuración
        avg_scores_df = pd.DataFrame(combination_scores).mean().to_dict()
        avg_scores_df['combination_name'] = config['name']
        all_results.append(avg_scores_df)
        
        print("\n  📊 Resultados promedio para la configuración:")
        for metric_name, score in avg_scores_df.items():
            if metric_name != 'combination_name':
                if pd.notna(score):
                    print(f"     - {metric_name}: {score:.4f}")
                else:
                    print(f"     - {metric_name}: Falló")

        if i < num_configs - 1:
            print("\n" + "-"*50)
            print("⏳ PAUSA EXTRA LARGA (300s) para la siguiente configuración...")
            print("-"*50)
            time.sleep(300)

    # Crear el DataFrame final
    final_results_df = pd.DataFrame(all_results)
    final_metric_cols = [name for name in metric_names if name in final_results_df.columns]
    final_results_df = final_results_df[['combination_name'] + final_metric_cols]
    return final_results_df
