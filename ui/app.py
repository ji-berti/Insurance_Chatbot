import streamlit as st
import requests

st.set_page_config(page_title="Asistente de Seguros", layout="centered")

# CHANGE THE URL FOR PRODUCTION
API_BASE_URL = "http://thirsty_ride:3000"

# Check backend API status
def check_api_status():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return res.status_code == 200
    except:
        return False

# System status (online or not)
is_online = check_api_status()

# Initialize history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Soy tu asistente de seguros. ¿En qué puedo ayudarte hoy?"}
    ]

# Sidebar: status, upload PDF, reset
with st.sidebar:
    st.markdown("## Sistema")
    st.markdown(
        f"**Estado del sistema:** {'Conectado 🟢' if is_online else 'Desconectado 🔴'}"
    )

    st.markdown("## 📄 Subir archivo PDF")

    uploaded_file = st.file_uploader("Selecciona un PDF", type="pdf")

    # Upload only if not uploaded yet
    if uploaded_file and "pdf_uploaded" not in st.session_state:
        with st.spinner("Subiendo archivo..."):
            response = requests.post(
                f"{API_BASE_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
            if response.status_code == 200:
                st.success("✅ Archivo cargado correctamente")
                st.session_state.pdf_uploaded = True
            else:
                st.error("❌ Error al subir el archivo")

    st.markdown("---")
    if st.button("Reiniciar conversación"):
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy tu asistente de seguros. ¿En qué puedo ayudarte hoy?"}
        ]
        st.rerun()

# Title and about session history
st.title("Asistente de Seguros")

st.markdown(
    """
    <div style="background-color:#f9f9f9; padding:10px; border-left:4px solid #f1c40f; border-radius:4px; font-size:14px;">
        ⚠️ <strong>Historial temporal:</strong> Esta conversación se guarda solo durante esta sesión.
        Si recargas la página o abres otra pestaña, el historial se perderá.
    </div>
    """,
    unsafe_allow_html=True
)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input field
if prompt := st.chat_input("Escribe tu mensaje..."):
    # Show user input
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Process response
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages if m["role"] in ("user", "assistant")
                ]
                response = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": prompt, "conversation_history": history},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if response.status_code == 200:
                    answer = response.json().get("answer", "⚠️ Respuesta vacía del servidor")
                else:
                    answer = "⚠️ Error al obtener respuesta del servidor"
            except Exception:
                answer = "❌ Ocurrió un error al contactar al asistente."

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
