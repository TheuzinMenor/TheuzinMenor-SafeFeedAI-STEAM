
import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download
st.set_page_config(
    page_title="SafeFeed AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, .stMarkdown {
    color: #cbd5e1;
}

[data-testid="stFileUploader"] {
    background: #111c30;
    border: 1px dashed #3b82f6;
    border-radius: 16px;
    padding: 16px;
}

.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #1d4ed8;
    color: white;
    border: 1px solid #60a5fa;
}

[data-testid="stMetric"] {
    background: #111c30;
    border: 1px solid #263449;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #60a5fa;
}

[data-testid="stProgressBar"] > div > div {
    background: #3b82f6;
}

hr {
    border-color: #263449;
}
</style>
""", unsafe_allow_html=True)


st.title("🛡️ SafeFeed AI")
st.subheader("Análise inteligente de vídeos")

st.write(
    "Envie um vídeo para verificar possíveis conteúdos "
    "gráficos (gore) ou adultos."
)

st.caption(
    "Análise experimental local. A IA pode cometer erros "
    "e não garante que todo o conteúdo impróprio seja detectado."
)

CAMINHO_MODELO = hf_hub_download(
    repo_id="OwenElliott/image-safety-classifier-xs",
    filename="onnx/image-safety-classifier-xs.onnx"
)


MAX_QUADROS = 30
LIMITE_ALERTA = 0.50

video = st.file_uploader(
    "Escolha um vídeo",
    type=["mp4", "mov", "avi", "mkv"]
)

if video is not None:
    st.video(video)

    if st.button("Analisar vídeo com IA"):
        caminho = None
        captura = None

        try:
            if not os.path.exists(CAMINHO_MODELO):
                st.error("Não encontrei o modelo de IA.")
                st.stop()

            with st.spinner("Carregando a IA..."):
                modelo = ort.InferenceSession(
                    CAMINHO_MODELO,
                    providers=["CPUExecutionProvider"]
                )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as arquivo:
                arquivo.write(video.getvalue())
                caminho = arquivo.name

            captura = cv2.VideoCapture(caminho)

            if not captura.isOpened():
                st.error("Não foi possível ler este vídeo.")
            else:
                total = int(
                    captura.get(cv2.CAP_PROP_FRAME_COUNT)
                )
                fps = captura.get(cv2.CAP_PROP_FPS)

                if total <= 0:
                    st.error("O vídeo não contém quadros legíveis.")
                else:
                    quantidade = min(MAX_QUADROS, total)

                    # Distribui os pontos de análise pelo vídeo inteiro.
                    indices = np.linspace(
                        0, total - 1,
                        num=quantidade,
                        dtype=int
                    )

                    st.write(f"Quadros no vídeo: {total}")
                    st.write(f"Quadros que serão analisados: {quantidade}")

                    barra = st.progress(0)
                    status = st.empty()
                    resultados = []

                    nomes = ["Gore (NSFL)", "Adulto (NSFW)", "Seguro (SFW)"]

                    for posicao, indice in enumerate(indices):
                        status.write(
                            f"Analisando quadro {posicao + 1} "
                            f"de {quantidade}..."
                        )

                        captura.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            int(indice)
                        )

                        sucesso, quadro = captura.read()

                        if sucesso:
                            rgb = cv2.cvtColor(
                                quadro, cv2.COLOR_BGR2RGB
                            )

                            imagem = Image.fromarray(rgb).convert(
                                "RGB"
                            ).resize(
                                (224, 224),
                                Image.Resampling.BILINEAR
                            )

                            pixels = np.asarray(
                                imagem, dtype=np.float32
                            ).transpose(2, 0, 1)[None, ...]

                            probabilidades = modelo.run(
                                None, {"image": pixels}
                            )[0][0]

                            resultados.append({
                                "indice": int(indice),
                                "imagem": rgb,
                                "probabilidades": probabilidades
                            })

                        barra.progress(
                            (posicao + 1) / quantidade
                        )

                    status.empty()
                    barra.empty()

                    if resultados:
                        st.success(
                            f"{len(resultados)} quadros analisados!"
                        )

                        maior_gore = max(
                            float(r["probabilidades"][0])
                            for r in resultados
                        )
                        maior_adulto = max(
                            float(r["probabilidades"][1])
                            for r in resultados
                        )

                        
                        st.subheader("📊 Relatório de segurança")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### 🩸 Detecção de gore")
                            st.metric(
                                "Maior probabilidade",
                                f"{maior_gore:.1%}"
                            )
                            st.progress(
                                min(max(maior_gore, 0.0), 1.0)
                            )

                            if maior_gore >= LIMITE_ALERTA:
                                st.error("⚠️ Possível gore detectado")
                            else:
                                st.success("Abaixo do limite de alerta")

                        with col2:
                            st.markdown("#### 🔞 Conteúdo adulto")
                            st.metric(
                                "Maior probabilidade",
                                f"{maior_adulto:.1%}"
                            )
                            st.progress(
                                min(max(maior_adulto, 0.0), 1.0)
                            )

                            if maior_adulto >= LIMITE_ALERTA:
                                st.warning("⚠️ Possível conteúdo adulto")
                            else:
                                st.success("Abaixo do limite de alerta")

                        st.divider()

                        st.markdown("### 🛡️ Conclusão da análise")

                        if (
                            maior_gore >= LIMITE_ALERTA
                            or maior_adulto >= LIMITE_ALERTA
                        ):
                            st.warning(
                                "O vídeo precisa de revisão humana. "
                                "A IA encontrou possíveis sinais de "
                                "conteúdo impróprio nos quadros analisados."
                            )
                        else:
                            st.info(
                                "Nenhum quadro analisado ultrapassou "
                                "o limite de alerta. Isso não significa "
                                "que o vídeo inteiro seja seguro."
                            )

                        st.caption(
                            f"Quadros analisados: {len(resultados)} "
                            f"de até {MAX_QUADROS}. "
                            "As probabilidades são estimativas do modelo, "
                            "não garantias de segurança."
                        )

                        
                        suspeitos = [
                            r for r in resultados
                            if (
                                float(r["probabilidades"][0])
                                >= LIMITE_ALERTA
                                or float(r["probabilidades"][1])
                                >= LIMITE_ALERTA
                            )
                        ]

                        outros = [
                            r for r in resultados
                            if r not in suspeitos
                        ]

                        st.subheader("🔎 Quadros para revisão")

                        if suspeitos:
                            st.warning(
                                f"{len(suspeitos)} quadro(s) "
                                "merecem revisão humana."
                            )

                            for numero, resultado in enumerate(
                                suspeitos, start=1
                            ):
                                p = resultado["probabilidades"]

                                segundo = (
                                    resultado["indice"] / fps
                                    if fps and fps > 0
                                    else None
                                )

                                legenda = f"🚨 Quadro suspeito {numero}"

                                if segundo is not None:
                                    legenda += f" — {segundo:.1f} s"

                                st.image(
                                    resultado["imagem"],
                                    caption=legenda
                                )

                                if float(p[0]) >= LIMITE_ALERTA:
                                    st.error(
                                        f"Possível gore: {p[0]:.1%}"
                                    )

                                if float(p[1]) >= LIMITE_ALERTA:
                                    st.warning(
                                        f"Possível conteúdo adulto: "
                                        f"{p[1]:.1%}"
                                    )

                                st.caption(
                                    f"Seguro: {p[2]:.1%}"
                                )

                        else:
                            st.success(
                                "Nenhum quadro ultrapassou o limite "
                                "de alerta. Isso não garante que "
                                "o vídeo inteiro seja seguro."
                            )

                        with st.expander(
                            f"Ver os outros quadros ({len(outros)})"
                        ):
                            for numero, resultado in enumerate(
                                outros, start=1
                            ):
                                p = resultado["probabilidades"]

                                segundo = (
                                    resultado["indice"] / fps
                                    if fps and fps > 0
                                    else None
                                )

                                legenda = f"Quadro sem alerta {numero}"

                                if segundo is not None:
                                    legenda += f" — {segundo:.1f} s"

                                st.image(
                                    resultado["imagem"],
                                    caption=legenda
                                )

                                st.write(
                                    f"Gore: {p[0]:.1%} · "
                                    f"Adulto: {p[1]:.1%} · "
                                    f"Seguro: {p[2]:.1%}"
                                )

                    else:
                        st.error(
                            "Não foi possível extrair quadros do vídeo."
                        )


        except Exception as erro:
            st.error(f"Erro durante a análise: {erro}")

        finally:
            if captura is not None:
                captura.release()

            if caminho is not None and os.path.exists(caminho):
                os.remove(caminho)
