import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
import pandas as pd
from datetime import datetime
import time

# ==========================================
# CONFIGURAÇÃO VISUAL
# ==========================================
st.set_page_config(page_title="Leitor V25 - Validação FH-40", page_icon="✅", layout="wide")

st.markdown("""
<style>
    div.stButton > button:first-child {
        height: 3.8em;
        font-size: 20px;
        font-weight: bold;
        border-radius: 12px;
        border: 1px solid #ddd;
    }
    div[data-testid="stFileUploader"] {
        width: 100%;
        padding: 10px;
        border: 2px dashed #4CAF50;
        border-radius: 10px;
    }
    .info-card {
        background-color: #e3f2fd;
        border-left: 6px solid #2196F3;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 8px;
    }
    .status-box {
        padding: 25px;
        text-align: center;
        border-radius: 15px;
        margin: 20px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .status-ok {
        background-color: #2e7d32; /* Verde Forte */
        border: 4px solid #1b5e20;
    }
    .status-nok {
        background-color: #c62828; /* Vermelho Forte */
        border: 4px solid #b71c1c;
    }
    .big-text {
        font-size: 35px;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .calc-text {
        font-size: 18px;
        margin-top: 10px;
        font-family: monospace;
        background: rgba(0,0,0,0.2);
        padding: 5px;
        border-radius: 5px;
        display: inline-block;
    }
    .ai-badge {
        background-color: #fce4ec;
        color: #880e4f;
        padding: 8px;
        border-radius: 6px;
        font-size: 14px;
        border: 1px solid #f8bbd0;
        margin-bottom: 15px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 0. DADOS TÉCNICOS (AGORA INCLUINDO FH-40)
# ==========================================
DADOS_TECNICOS = {
    "FH-40": {
        "Valor Padrão": "200 nSv/h",
        "Tolerância": "± 10%",
        "Faixa Aceitável": "180 - 220 nSv/h",
        "Status": "Monitoramento Ativo"
    },
    "Fixo na maleta VacuTec 70043- A": {
        "Nº de série": "99002",
        "Ref. Esperada": "1,39 uSv/h (1,25 - 1,53)",
        "Erro aceitável": "± 10%",
        "Carregador": "Universal 15 V DC - 6 A"
    },
    "Portátil VacuTec 70043- A": {
        "Nº de série": "00014",
        "Ref. Esperada": "2,88 uSv/h (2,59 - 3,17)",
        "Erro aceitável": "± 10%",
        "Carregador": "PUP55-13 Protek + adaptador"
    },
    "Portátil VacuTec 70046- A": {
        "Nº de série": "1800124",
        "Ref. Esperada": "3,07 uSv/h (2,76 - 3,38)",
        "Erro aceitável": "± 10%",
        "Carregador": "PUP55-13 Protek + adaptador"
    }
}

# ==========================================
# 1. ESTADO DA SESSÃO
# ==========================================
if 'buffer_leituras' not in st.session_state: st.session_state.buffer_leituras = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'equipamento_selecionado' not in st.session_state: st.session_state.equipamento_selecionado = None
if 'modo_vacutec' not in st.session_state: st.session_state.modo_vacutec = False

# ==========================================
# 2. FUNÇÕES DE IA E VISÃO
# ==========================================
ARQUIVO_CSV = "leituras_equipamentos.csv"

def salvar_leitura(modelo_equip, valor, tipo="Única"):
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    novo_dado = pd.DataFrame([{"Data": data_hora, "Equipamento": modelo_equip, "Leitura": int(valor), "Tipo": tipo}])
    if not os.path.exists(ARQUIVO_CSV): novo_dado.to_csv(ARQUIVO_CSV, index=False)
    else: novo_dado.to_csv(ARQUIVO_CSV, mode='a', header=False, index=False)
    return True

# --- FUNÇÃO DE VALIDAÇÃO DE FAIXA ---
def verificar_faixa_fh40(valor_lido):
    alvo = 200
    tolerancia = 0.10 # 10%
    minimo = int(alvo * (1 - tolerancia)) # 180
    maximo = int(alvo * (1 + tolerancia)) # 220
    
    dentro = minimo <= valor_lido <= maximo
    return dentro, minimo, maximo

# --- PROCESSAMENTO DE IMAGEM ---
def aplicar_clahe(img_gray, clip_limit=3.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)

def unir_fragmentos_inteligente(caixas, max_dist_y):
    if not caixas: return []
    caixas.sort(key=lambda c: c[0])
    merged = []
    curr_x, curr_y, curr_w, curr_h = caixas[0]
    for i in range(1, len(caixas)):
        next_x, next_y, next_w, next_h = caixas[i]
        sobrepoe_x = next_x < (curr_x + curr_w + 5)
        dist_vertical = abs(curr_y - next_y) 
        if sobrepoe_x and dist_vertical <= max_dist_y:
            novo_x = min(curr_x, next_x); novo_y = min(curr_y, next_y)
            novo_max_x = max(curr_x + curr_w, next_x + next_w); novo_max_y = max(curr_y + curr_h, next_y + next_h)
            curr_x, curr_y, curr_w, curr_h = novo_x, novo_y, novo_max_x - novo_x, novo_max_y - novo_y
        else:
            merged.append((curr_x, curr_y, curr_w, curr_h))
            curr_x, curr_y, curr_w, curr_h = next_x, next_y, next_w, next_h
    merged.append((curr_x, curr_y, curr_w, curr_h))
    return merged

def filtrar_outliers_horizontais(caixas, max_gap_x):
    if not caixas: return []
    caixas.sort(key=lambda c: c[0])
    grupos = []
    grupo_atual = [caixas[0]]
    for i in range(1, len(caixas)):
        box_anterior = caixas[i-1]; box_atual = caixas[i]
        if (box_atual[0] - (box_anterior[0] + box_anterior[2])) > max_gap_x:
            grupos.append(grupo_atual); grupo_atual = [box_atual]
        else: grupo_atual.append(box_atual)
    grupos.append(grupo_atual)
    return max(grupos, key=lambda g: sum(box[2] for box in g))

def separar_digitos_colados(caixas):
    novas_caixas = []
    for (x, y, w, h) in caixas:
        ratio = w / float(h)
        if ratio > 0.85: 
            metade_w = w // 2
            novas_caixas.append((x, y, metade_w, h)); novas_caixas.append((x + metade_w, y, metade_w, h))
        else: novas_caixas.append((x, y, w, h))
    return novas_caixas

def segmentar_digitos(img_recortada_pil, params):
    img_cv = np.array(img_recortada_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    if params.get('usar_clahe', True):
        gray = aplicar_clahe(gray, clip_limit=params['clahe_clip'])
    elif params['contraste_alto'] > 1.0:
        gray = cv2.convertScaleAbs(gray, alpha=params['contraste_alto'], beta=0)

    k = params['blur_k'] | 1 
    gray_blurred = cv2.GaussianBlur(gray, (k, k), 0)
    metodo = cv2.THRESH_BINARY_INV if params['inverter_bin'] else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray_blurred, params['thresh_val'], 255, metodo)
    
    if params.get('dilatacao_iter', 0) > 0:
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=params['dilatacao_iter'])

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caixas_brutas = []
    area_total = gray.shape[0] * gray.shape[1]
    min_area_perc = params.get('area_min_perc', 0.1) / 100.0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w*h) > (area_total * min_area_perc): caixas_brutas.append((x, y, w, h))

    caixas_mescladas = unir_fragmentos_inteligente(caixas_brutas, params['max_dist_y'])
    caixas_grupo = filtrar_outliers_horizontais(caixas_mescladas, params['max_gap_x'])
    caixas_separadas = separar_digitos_colados(caixas_grupo)

    if caixas_separadas:
        max_h = max(c[3] for c in caixas_separadas)
        caixas_finais = [c for c in caixas_separadas if c[3] > (max_h * 0.40)]
    else: caixas_finais = []

    if len(caixas_finais) > 3:
        caixas_finais.sort(key=lambda c: c[0])
        caixas_finais = caixas_finais[:3]

    img_debug = gray.copy() 
    img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2RGB)
    recortes = []
    margin = params['margem']
    h_img, w_img = gray.shape

    for (x, y, w, h) in caixas_finais:
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        y1, y2 = max(0, y-margin), min(h_img, y+h+margin)
        x1, x2 = max(0, x-margin), min(w_img, x+w+margin)
        recortes.append(Image.fromarray(img_cv[y1:y2, x1:x2]))

    return recortes, img_debug, thresh

def processar_para_yolo_digits(img_pil):
    img_gray = img_pil.convert("L")
    if np.mean(np.array(img_gray)) < 127: img_gray = ImageOps.invert(img_gray)
    img_rgb = img_gray.convert("RGB")
    largura, altura = img_rgb.size
    novo = max(largura, altura)
    img = Image.new("RGB", (novo, novo), (255, 255, 255))
    img.paste(img_rgb, ((novo-largura)//2, (novo-altura)//2))
    return img

def detectar_e_recortar_tela(img_pil, model_det, conf, crop_fraction):
    results = model_det(img_pil, conf=conf)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return img_pil, False, None, 0.0
    
    box_data = results[0].boxes[0]
    box = box_data.xyxy[0].cpu().numpy()
    class_id = int(box_data.cls[0].item())
    class_name = results[0].names[class_id]
    confidence = box_data.conf[0].item()
    
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    cx, cy = x1+w/2, y1+h/2
    nw, nh = w*crop_fraction, h*crop_fraction
    
    crop = img_pil.crop((max(0, cx-nw/2), max(0, cy-nh/2), min(img_pil.width, cx+nw/2), min(img_pil.height, cy+nh/2)))
    return crop, True, class_name, confidence

# ==========================================
# 3. SETUP MODELOS
# ==========================================
PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
NOME_PADRAO_TELA = "modelo_de_tela.pt"
NOME_PADRAO_NUMERO = "modelo_de_numero.pt"
arquivos_pt = [f for f in os.listdir(PASTA_ATUAL) if f.endswith('.pt')]

idx_tela, idx_num = 0, 0
if NOME_PADRAO_TELA in arquivos_pt: idx_tela = arquivos_pt.index(NOME_PADRAO_TELA)
if NOME_PADRAO_NUMERO in arquivos_pt: idx_num = arquivos_pt.index(NOME_PADRAO_NUMERO)
elif len(arquivos_pt) > 1: idx_num = 1

st.sidebar.header("⚙️ Área Técnica")
usar_clahe = st.sidebar.checkbox("Anti-Reflexo", value=True)
clahe_force = st.sidebar.slider("Força CLAHE", 1.0, 8.0, 3.0)
thresh = st.sidebar.slider("Limiar", 0, 255, 15)
dilatacao_iter = st.sidebar.slider("Engrossar", 0, 5, 2)
model_det_name = st.sidebar.selectbox("Modelo Tela", arquivos_pt, index=idx_tela)
model_cls_name = st.sidebar.selectbox("Modelo Digito", arquivos_pt, index=idx_num)

st.sidebar.markdown("---")
st.sidebar.info("**ℹ️ IA YOLOv8 Ativada**")

@st.cache_resource
def load_model(p): return YOLO(os.path.join(PASTA_ATUAL, p))
try:
    detector = load_model(model_det_name); classificador = load_model(model_cls_name)
except: st.error("Erro carregando modelos."); st.stop()

params = {
    'contraste_alto': 1.5, 'usar_norm': False, 'inverter_bin': True,
    'blur_k': 3, 'thresh_val': thresh, 'margem': 2,
    'max_dist_y': 100, 'max_gap_x': 100, 
    'dilatacao_iter': dilatacao_iter, 'area_min_perc': 0.001,
    'usar_clahe': usar_clahe, 'clahe_clip': clahe_force
}

# ==========================================
# 4. SUPER SELETOR
# ==========================================
st.markdown("### 1. Selecione o Equipamento")

col_e1, col_e2, col_e3 = st.columns(3)
if col_e1.button("FH-40", use_container_width=True):
    st.session_state.equipamento_selecionado = "FH-40"
    st.session_state.modo_vacutec = False
    st.rerun()

if col_e2.button("VacuTec ▾", use_container_width=True):
    st.session_state.modo_vacutec = True
    st.session_state.equipamento_selecionado = None
    st.rerun()

if col_e3.button("Ludlum", use_container_width=True):
    st.session_state.equipamento_selecionado = "Ludlum"
    st.session_state.modo_vacutec = False
    st.rerun()

if st.session_state.modo_vacutec:
    st.info("📂 Qual modelo VacuTec?")
    sub_c1, sub_c2, sub_c3 = st.columns(3)
    if sub_c1.button("Fixo na maleta\n70043- A", use_container_width=True):
        st.session_state.equipamento_selecionado = "Fixo na maleta VacuTec 70043- A"; st.rerun()
    if sub_c2.button("Portátil\n70043- A", use_container_width=True):
        st.session_state.equipamento_selecionado = "Portátil VacuTec 70043- A"; st.rerun()
    if sub_c3.button("Portátil\n70046- A", use_container_width=True):
        st.session_state.equipamento_selecionado = "Portátil VacuTec 70046- A"; st.rerun()

st.divider()

if st.session_state.equipamento_selecionado:
    nome_eq = st.session_state.equipamento_selecionado
    st.markdown(f"""<div style="background-color:#4CAF50; color:white; padding:10px; border-radius:5px; text-align:center; font-size:20px; font-weight:bold; margin-bottom: 20px;">✅ {nome_eq}</div>""", unsafe_allow_html=True)
    if nome_eq in DADOS_TECNICOS:
        dados = DADOS_TECNICOS[nome_eq]
        linhas_html = "".join([f"<div class='info-row'><span style='font-weight:bold;'>{k}:</span><span style='color:{'#d32f2f' if 'Ref' in k else '#0277bd'}; font-weight:bold;'>{v}</span></div>" for k, v in dados.items()])
        st.markdown(f"<div class='info-card'><div style='color:#0d47a1; font-weight:bold; font-size:18px; margin-bottom:10px;'>ℹ️ Especificações</div>{linhas_html}</div>", unsafe_allow_html=True)
else:
    st.markdown("""<div style="background-color:#f44336; color:white; padding:10px; border-radius:5px; text-align:center; margin-bottom: 20px;">⚠️ NENHUM EQUIPAMENTO SELECIONADO</div>""", unsafe_allow_html=True)

# ==========================================
# 5. INPUTS E PROCESSAMENTO
# ==========================================
st.markdown("### 2. Captura")
cam_input = st.camera_input("📸 Câmera Aberta", label_visibility="collapsed")
st.markdown("**Ou carregue um arquivo:**")
file_input = st.file_uploader("Upload", type=['jpg','png','jpeg'], key=f"up_{st.session_state.uploader_key}", label_visibility="collapsed")

img_processar = None
origem = ""
if cam_input: img_processar = Image.open(cam_input); origem = "camera"
elif file_input: img_processar = Image.open(file_input); origem = "upload"

if img_processar:
    st.markdown("---")
    
    # Detecção e Auto-Seleção FH-40
    img_tela, achou, detected_class, conf = detectar_e_recortar_tela(img_processar, detector, 0.15, 1.0)
    
    if achou and st.session_state.equipamento_selecionado is None and conf > 0.6:
        st.session_state.equipamento_selecionado = "FH-40"
        st.session_state.modo_vacutec = False
        st.toast(f"🤖 IA Reconheceu: {detected_class}. Selecionando FH-40...", icon="🧠")
        time.sleep(0.5)
        st.rerun()

    if not st.session_state.equipamento_selecionado:
        st.error("🛑 SELECIONE UM EQUIPAMENTO PARA CONTINUAR")
    else:
        st.markdown(f"""<div class="ai-badge">🤖 IA YOLOv8 | Objeto: '{detected_class}' | Confiança: {conf:.1%}</div>""", unsafe_allow_html=True)
        
        with st.spinner("🤖 Lendo..."):
            recortes, img_debug, thresh_img = segmentar_digitos(img_tela, params)
            leitura_str = ""
            for r in recortes:
                im = processar_para_yolo_digits(r)
                res = classificador(im, verbose=False)
                leitura_str += res[0].names[res[0].probs.top1]
            
            # Exibição de Imagens
            c1, c2, c3 = st.columns(3)
            c1.image(img_tela, caption="Recorte")
            c2.image(img_debug, caption="Visão Robô")
            c3.image(thresh_img, caption="Contraste")

            valor_lido = None
            try:
                limpo = "".join(filter(str.isdigit, leitura_str))
                if limpo:
                    valor_lido = int(limpo)
                    
                    # -----------------------------------------------
                    # LÓGICA DE VALIDAÇÃO VISUAL (GREEN BOX)
                    # -----------------------------------------------
                    if st.session_state.equipamento_selecionado == "FH-40":
                        aprovado, mn, mx = verificar_faixa_fh40(valor_lido)
                        
                        if aprovado:
                            html_status = f"""
                            <div class="status-box status-ok">
                                <div class="big-text">✅ APROVADO</div>
                                <div class="calc-text">
                                    Valor: {valor_lido} nSv/h<br>
                                    Faixa Permitida: {mn} a {mx} nSv/h (±10%)
                                </div>
                            </div>
                            """
                        else:
                            html_status = f"""
                            <div class="status-box status-nok">
                                <div class="big-text">🚫 REPROVADO</div>
                                <div class="calc-text">
                                    Valor: {valor_lido} nSv/h<br>
                                    FORA DA FAIXA: {mn} a {mx} nSv/h
                                </div>
                            </div>
                            """
                        st.markdown(html_status, unsafe_allow_html=True)
                    else:
                        # Exibição padrão para outros equipamentos
                        st.markdown(f"""
                        <div style="text-align:center; margin: 20px 0;">
                            <span style="font-size: 60px; font-weight: bold; color: #2196F3;">{valor_lido}</span>
                            <br><span style="color: gray;">LIDO</span>
                        </div>
                        """, unsafe_allow_html=True)

                else: st.warning("⚠️ Números não identificados.")
            except: st.error("Erro.")

            if valor_lido is not None:
                b1, b2, b3 = st.columns(3)
                if b1.button("✅ SALVAR", type="primary", use_container_width=True):
                    salvar_leitura(st.session_state.equipamento_selecionado, valor_lido, "Única")
                    st.success("SALVO!")
                    if origem == "upload": st.session_state.uploader_key += 1
                    time.sleep(1); st.rerun()
                
                txt_med = f"➕ SOMAR ({len(st.session_state.buffer_leituras)})" if st.session_state.buffer_leituras else "➕ NA MÉDIA"
                if b2.button(txt_med, use_container_width=True):
                    st.session_state.buffer_leituras.append(valor_lido)
                    if origem == "upload": st.session_state.uploader_key += 1
                    st.rerun()
                    
                if b3.button("🗑️ DESCARTAR", use_container_width=True):
                    if origem == "upload": st.session_state.uploader_key += 1
                    st.rerun()

if st.session_state.buffer_leituras:
    st.markdown("---")
    media = int(sum(st.session_state.buffer_leituras)/len(st.session_state.buffer_leituras))
    if st.button(f"💾 FECHAR MÉDIA: {media}", type="primary", use_container_width=True):
        salvar_leitura(st.session_state.equipamento_selecionado, media, "Média")
        st.session_state.buffer_leituras = []; st.balloons(); st.rerun()
