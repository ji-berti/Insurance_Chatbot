import streamlit as st
import requests
import uuid

st.set_page_config(page_title="Insurance Chatbot", layout="centered", page_icon="🤖")

API_BASE_URL = "http://api:3000"

# --- Session management ---
# Initialize session state if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    # Initialize file processing state
    st.session_state.file_processed_for_session = False
    st.session_state.uploaded_file_name = None # Remember the file name for the session

# Check backend API status
def check_api_status():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return res.status_code == 200
    except requests.exceptions.RequestException:
        return False

is_online = check_api_status()

# Initialize history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "¡Hola! Soy tu asistente de seguros. ¿En qué puedo ayudarte hoy?"}
    ]

# Sidebar: status, upload PDF, reset
with st.sidebar:
    st.markdown("## Sistema")
    st.markdown(
        f"**Estado:** {'`Conectado`' if is_online else '`Desconectado`'}"
    )

    st.markdown("## 📄 Subir archivo PDF")

    # Uploader key depends on session_id to ensure unique file upload per session
    uploaded_file = st.file_uploader(
        "Selecciona un PDF para esta sesión", type="pdf", key=f"uploader_{st.session_state.session_id}"
    )

    # Logic to handle file upload and processing
    if uploaded_file is None:
        st.session_state.file_processed_for_session = False
        st.session_state.uploaded_file_name = None
    elif uploaded_file and (not st.session_state.file_processed_for_session or st.session_state.uploaded_file_name != uploaded_file.name):
        # Proccess the file only if it hasn't been processed for this session or if it's a new file
        with st.spinner("Procesando archivo para la sesión..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            data = {"session_id": st.session_state.session_id}

            try:
                response = requests.post(
                    f"{API_BASE_URL}/upload",
                    files=files,
                    data=data
                )
                if response.status_code == 200:
                    st.success("✅ Archivo procesado para esta sesión.")
                    st.info("Ahora puedes hacer preguntas sobre este documento.")
                    st.session_state.file_processed_for_session = True # Mark as processed
                    st.session_state.uploaded_file_name = uploaded_file.name # Save the name of the processed file
                else:
                    st.error(f"❌ Error al procesar: {response.text}")
                    st.session_state.file_processed_for_session = False # If there's an error, allow retry
                    st.session_state.uploaded_file_name = None
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error de conexión: {e}")
                st.session_state.file_processed_for_session = False # If there's an error, allow retry
                st.session_state.uploaded_file_name = None
    elif st.session_state.file_processed_for_session and st.session_state.uploaded_file_name == uploaded_file.name:
        # Tell the user that the file has already been processed for this session
        st.info(f"El archivo '{uploaded_file.name}' ya ha sido procesado para esta sesión.")

    st.markdown("---")
    # Restart session button
    if st.button("Reiniciar Sesión"):
        st.session_state.session_id = str(uuid.uuid4()) # Generate new session ID
        st.session_state.messages = [
            {"role": "ai", "content": "He iniciado una nueva sesión. ¿Cómo puedo ayudarte?"}
        ]
        # Restart files states after session reset
        st.session_state.file_processed_for_session = False
        st.session_state.uploaded_file_name = None
        st.rerun()

# --- Chat main interface ---
st.title("Asistente de Seguros")
st.warning('⚠️ **Historial temporal:** La conversación y los archivos subidos se pierden al reiniciar la sesión o recargar la página.')

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu mensaje..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                payload = {
                    "question": prompt,
                    "conversation_history": history,
                    "session_id": st.session_state.session_id
                }

                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json=payload, 
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if response.status_code == 200:
                    answer = response.json().get("answer", "⚠️ Respuesta vacía del servidor")
                else:
                    answer = f"⚠️ Error del servidor: {response.text}"
            except Exception as e:
                answer = f"❌ Ocurrió un error al contactar al asistente: {e}"

            st.markdown(answer)
            st.session_state.messages.append({"role": "ai", "content": answer})