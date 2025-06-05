import streamlit as st
import requests
import json

# * BORRAR DEBUGS ANTES DE PRODUCCIÓN *

st.set_page_config(page_title="Chat de Seguros - Historial Temporal", layout="centered")

# Inicializar el historial de conversación en sesión
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
    st.session_state.initial_message = (
        "¡Hola! Soy tu asistente de seguros. Puedo ayudarte con preguntas sobre pólizas, "
        "coberturas y más. Tu historial de conversación se mantiene durante esta sesión."
    )
    st.session_state.conversation_history.append({"role": "ai", "content": st.session_state.initial_message})

# Cambiar por el nombre del devcontainer del API (en desarrollo)
# Cambiar por el nombre del servicio del API en producción (docker compose)
API_BASE_URL = "http://tender_boyd:3000" 

st.title("💬 Chat de Seguros - Historial Temporal")

with st.container():
    st.markdown(
        """
        <div style="background-color:#d4edda; padding:10px; border-radius:5px; color:#155724; font-size:14px;">
            📝 <strong>Historial en memoria:</strong> Tu conversación se mantiene activa solo durante esta sesión.
            Si recargas la página o abres otra pestaña, empezarás una conversación nueva e independiente.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Entrada del usuario (movida arriba para procesar antes de mostrar mensajes)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Escribe tu pregunta sobre seguros...", key="user_input", placeholder="Ej. ¿Qué cubre mi póliza?")
    submit_button = st.form_submit_button("Enviar")

# Procesar envío ANTES de mostrar los mensajes
if submit_button and user_input.strip():
    # Agregar pregunta del usuario al historial
    st.session_state.conversation_history.append({"role": "human", "content": user_input})

    with st.spinner("🤔 Pensando..."):
        try:
            # Preparar el historial para enviar (excluyendo el mensaje actual del usuario)
            history_for_api = st.session_state.conversation_history[:-1]
            
            # Debug: mostrar lo que se está enviando
            st.write("🔍 **Debug - Enviando a API:**")
            st.write(f"Pregunta: {user_input}")
            st.write(f"Historial: {len(history_for_api)} mensajes")
            
            response = requests.post(
                f"{API_BASE_URL}/ask",
                json={
                    "question": user_input,
                    "conversation_history": history_for_api,
                },
                timeout=30,  # Aumentar timeout
                headers={"Content-Type": "application/json"}
            )
            
            # Debug: mostrar respuesta HTTP
            st.write(f"🔍 **Debug - Respuesta HTTP:** Status {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    st.write(f"🔍 **Debug - JSON recibido:** {data}")
                    
                    # Extraer la respuesta del chatbot
                    bot_response = data.get("answer", "⚠️ Respuesta vacía del servidor")
                    if not bot_response:
                        bot_response = "⚠️ El servidor no devolvió una respuesta"
                    
                    st.session_state.conversation_history.append({"role": "ai", "content": bot_response})
                    st.write(f"✅ **Respuesta agregada al historial**")
                    
                except json.JSONDecodeError as je:
                    error_msg = f"❌ Error al decodificar JSON: {str(je)}\nRespuesta raw: {response.text[:500]}"
                    st.session_state.conversation_history.append({"role": "ai", "content": error_msg})
                    st.error(error_msg)
            else:
                error_msg = f"❌ Error HTTP {response.status_code}: {response.text[:500]}"
                st.session_state.conversation_history.append({"role": "ai", "content": error_msg})
                st.error(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "⏰ La solicitud tardó demasiado. El servidor puede estar sobrecargado."
            st.session_state.conversation_history.append({"role": "ai", "content": error_msg})
            st.error(error_msg)
            
        except requests.exceptions.ConnectionError:
            error_msg = f"🔌 No se pudo conectar al servidor en {API_BASE_URL}"
            st.session_state.conversation_history.append({"role": "ai", "content": error_msg})
            st.error(error_msg)
            
        except Exception as e:
            error_msg = f"❌ Error inesperado: {str(e)}\nTipo: {type(e).__name__}"
            st.session_state.conversation_history.append({"role": "ai", "content": error_msg})
            st.error(error_msg)

# Mostrar mensajes del historial (después del procesamiento)
st.markdown("---")
st.subheader("💬 Conversación")

chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "human":
            st.markdown(
                f"""
                <div style="text-align:right; background-color:#007bff; color:white; padding:10px;
                            border-radius:10px; margin:5px 0; max-width:80%; margin-left:auto;">
                    <strong>Tú:</strong> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="text-align:left; background-color:#e9ecef; color:#333; padding:10px;
                            border-radius:10px; margin:5px 0; max-width:80%; margin-right:auto;">
                    <strong>🤖 Asistente:</strong> {msg['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )

# Debug: mostrar estado actual
st.sidebar.title("🔍 Debug Info")
st.sidebar.write(f"**Total mensajes:** {len(st.session_state.conversation_history)}")
st.sidebar.write(f"**Último mensaje:** {st.session_state.conversation_history[-1]['role'] if st.session_state.conversation_history else 'N/A'}")

# Mostrar historial completo en sidebar para debug
if st.sidebar.checkbox("Mostrar historial completo"):
    st.sidebar.json(st.session_state.conversation_history)

# Botones de control
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Limpiar conversación"):
        st.session_state.conversation_history = []
        st.session_state.conversation_history.append({"role": "ai", "content": st.session_state.initial_message})
        st.rerun()

with col2:
    if st.button("🔄 Refrescar"):
        st.rerun()

# Test de conectividad
st.markdown("---")
if st.button("🔧 Probar conexión con API"):
    try:
        test_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if test_response.status_code == 200:
            st.success("✅ Conexión exitosa con la API")
        else:
            st.error(f"❌ API respondió con código {test_response.status_code}")
    except Exception as e:
        st.error(f"❌ Error de conexión: {str(e)}")