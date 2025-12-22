import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
import pandas as pd
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="Leitor V18 - Câmera Nativa & Upload", page_icon="📸", layout="wide")

# ==========================================
# 0. ESTADO DA SESSÃO
# ==========================================
if 'buffer_leituras' not in st.session_state:
    st.session_state.buffer_leituras = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# ==========================================
# 1. FUNÇÕES DE BANCO DE DADOS (CSV)
# ==========================================
ARQUIVO_CSV = "leituras_equipamentos.csv"

def salvar_leitura(modelo_equip, valor, tipo="Única"):
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    novo_dado = pd.DataFrame([{
        "Data": data_hora,
        "Equipamento": modelo_equip,
        "Leitura": int(valor),
        "Tipo": tipo
    }])
    
    if not os.path.exists(ARQUIVO_CSV):
        novo_dado.to_csv(ARQUIVO_CSV, index=False)
    else:
        novo_dado.to_csv(ARQUIVO_CSV, mode='a', header=False, index=False)
    return True

# ==========================================
# 2. PROCESSAMENTO VISUAL (Seu código ajustado)
# ==========================================

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
            novo_x = min(curr_x, next_x)
            novo_y = min(curr_y, next_y)
            novo_max_x = max(curr_x + curr_w, next_x + next_w)
            novo_max_y = max(curr_y + curr_h, next_y + next_h)
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
        box_anterior = caixas[i-1]
        box_atual = caixas[i]
        if (box_atual[0] - (box_anterior[0] + box_anterior[2])) > max_gap_x:
            grupos.append(grupo_atual)
            grupo_atual = [box_atual]
        else:
            grupo_atual.append(box_atual)
    grupos.append(grupo_atual)
    return max(grupos, key=lambda g: sum(box[2] for box in g))

def separar_digitos_colados(caixas):
    novas_caixas = []
    for (x, y, w, h) in caixas:
        ratio = w / float(h)
        if ratio > 0.85: 
            metade_w = w // 2
            novas_caixas.append((x, y, metade_w, h))
            novas_caixas.append((x + metade_w, y, metade_w, h))
        else:
            novas_caixas.append((x, y, w, h))
    return novas_caixas

def segmentar_digitos(img_recortada_pil, params):
    img_cv = np.array(img_recortada_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    if params['contraste_alto'] > 1.0:
        gray = cv2.convertScaleAbs(gray, alpha=params['contraste_alto'], beta=0)

    k = params['blur_k'] | 1 
    gray_blurred = cv2.GaussianBlur(gray, (k, k), 0)
    metodo = cv2.THRESH_BINARY_INV if params['inverter_bin'] else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray_blurred, params['thresh_val'], 255, metodo)
    
    iteracoes = params.get('dilatacao_iter', 0)
    if iteracoes > 0:
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=iteracoes)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    caixas_brutas = []
    area_total = gray.shape[0] * gray.shape[1]
    
    min_area_perc = params.get('area_min_perc', 0.1) / 100.0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w*h) > (area_total * min_area_perc): 
            caixas_brutas.append((x, y, w, h))

    caixas_mescladas = unir_fragmentos_inteligente(caixas_brutas, params['max_dist_y'])
    caixas_grupo = filtrar_outliers_horizontais(caixas_mescladas, params['max_gap_x'])
    caixas_separadas = separar_digitos_colados(caixas_grupo)

    if caixas_separadas:
        max_h = max(c[3] for c in caixas_separadas)
        caixas_finais = [c for c in caixas_separadas if c[3] > (max_h * 0.40)]
    else:
        caixas_finais = []

    if len(caixas_finais) > 3:
        caixas_finais.sort(key=lambda c: c[0])
        caixas_finais = caixas_finais[:3]

    img_debug = img_cv.copy()
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
        return img_pil, False
    
    box = results[0].boxes[0].xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    cx, cy = x1+w/2, y1+h/2
    nw, nh = w*crop_fraction, h*crop_fraction
    return img_pil.crop((max(0, cx-nw/2), max(0, cy-nh/2), min(img_pil.width, cx+nw/2), min(img_pil.height, cy+nh/2))), True

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

st.sidebar.header("Modelos")
model_det_name = st.sidebar.selectbox("Detector Tela", arquivos_pt, index=idx_tela)
model_cls_name = st.sidebar.selectbox("Classificador", arquivos_pt, index=idx_num)

st.sidebar.divider()
st.sidebar.markdown("**Ajustes Finos**")
thresh = st.sidebar.slider("Limiar (Fixo: 15)", 0, 255, 15)
dilatacao_iter = st.sidebar.slider("Engrossar Dígitos", 0, 5, 2)
area_min_perc = st.sidebar.slider("Área Mínima (%)", 0.01, 1.0, 0.10)
conf_det = st.sidebar.slider("Sensibilidade Tela", 0.1, 1.0, 0.15)

@st.cache_resource
def load_model(p): return YOLO(os.path.join(PASTA_ATUAL, p))

try:
    detector = load_model(model_det_name)
    classificador = load_model(model_cls_name)
except: st.error("Erro carregando modelos."); st.stop()

params = {
    'contraste_alto': 1.5, 'usar_norm': False, 'inverter_bin': True,
    'blur_k': 3, 'thresh_val': thresh, 'margem': 2,
    'max_dist_y': 100, 'max_gap_x': 100, 
    'dilatacao_iter': dilatacao_iter, 'area_min_perc': area_min_perc
}

# ==========================================
# 4. INTERFACE UNIFICADA
# ==========================================
st.title("📸 Leitor V18: Câmera Nativa & Upload")

col_nome, col_info = st.columns([2, 1])
with col_nome:
    equipamento_atual = st.text_input("Nome do Equipamento:", placeholder="Ex: Pressostato-B1")
with col_info:
    qtd = len(st.session_state.buffer_leituras)
    if qtd > 0:
        m_parcial = int(sum(st.session_state.buffer_leituras)/qtd)
        st.metric("Média Parcial", f"{m_parcial}", delta=f"{qtd} leituras")
    else:
        st.metric("Status", "Aguardando", delta="0 leituras")

# Abas de Entrada
aba_upload, aba_camera = st.tabs(["📁 Upload de Arquivo", "📸 Câmera Direta"])

img_processar = None
origem = ""

# --- ABA 1: UPLOAD ---
with aba_upload:
    file = st.file_uploader("Arraste uma foto aqui", type=['jpg','png','jpeg'], key=f"up_{st.session_state.uploader_key}")
    if file:
        img_processar = Image.open(file)
        origem = "upload"

# --- ABA 2: CÂMERA (CORRIGIDO) ---
with aba_camera:
    cam_file = st.camera_input("Tirar Foto Agora")
    if cam_file:
        img_processar = Image.open(cam_file)
        origem = "camera"

# ==========================================
# 5. LÓGICA DE PROCESSAMENTO E DECISÃO
# ==========================================
if img_processar:
    st.divider()
    
    # 1. Detectar
    img_tela, achou = detectar_e_recortar_tela(img_processar, detector, conf_det, 1.0)
    
    # 2. Ler
    recortes, img_debug, _ = segmentar_digitos(img_tela, params)
    
    leitura_str = ""
    for r in recortes:
        im = processar_para_yolo_digits(r)
        res = classificador(im, verbose=False)
        leitura_str += res[0].names[res[0].probs.top1]
    
    # Visualização
    c1, c2 = st.columns(2)
    c1.image(img_tela, caption="Recorte da Tela", width=300)
    c2.image(img_debug, caption="Segmentação (Visão Robô)", width=300)

    # Validar
    valor_lido = None
    try:
        limpo = "".join(filter(str.isdigit, leitura_str))
        if limpo:
            valor_lido = int(limpo)
            st.markdown(f"<div style='text-align:center; background-color:#e8f5e9; padding:10px; border-radius:10px;'><h2 style='color:#2e7d32; margin:0;'>Leitura: {valor_lido}</h2></div>", unsafe_allow_html=True)
        else:
            st.warning("Não consegui identificar números.")
    except:
        st.error("Erro ao converter número.")

    # Botões de Decisão
    if valor_lido is not None:
        st.write("")
        b1, b2, b3 = st.columns(3)
        
        # A. Salvar Único
        if len(st.session_state.buffer_leituras) == 0:
            if b1.button("✅ Salvar Único e Próximo", type="primary", use_container_width=True):
                if not equipamento_atual:
                    st.toast("Preencha o nome do equipamento!")
                else:
                    salvar_leitura(equipamento_atual, valor_lido, "Única")
                    st.success("Salvo!")
                    st.session_state.uploader_key += 1 # Limpa upload
                    st.rerun()

        # B. Adicionar à Média
        texto_media = "➕ Iniciar Média" if len(st.session_state.buffer_leituras) == 0 else f"➕ Adicionar ({valor_lido}) à Média"
        if b2.button(texto_media, use_container_width=True):
            st.session_state.buffer_leituras.append(valor_lido)
            st.toast(f"Valor {valor_lido} adicionado!")
            # Se veio do upload, limpamos para forçar usuario a por outro
            if origem == "upload":
                st.session_state.uploader_key += 1
            st.rerun()
            
        # C. Tentar de novo
        if b3.button("🗑️ Descartar/Tentar de Novo", use_container_width=True):
            if origem == "upload":
                st.session_state.uploader_key += 1
            st.rerun()

# ==========================================
# 6. RODAPÉ (FINALIZAR MÉDIA)
# ==========================================
if st.session_state.buffer_leituras:
    st.markdown("---")
    st.markdown(f"### 📊 Finalizar '{equipamento_atual}'")
    st.write(f"Valores coletados: {st.session_state.buffer_leituras}")
    
    media_final = int(sum(st.session_state.buffer_leituras) / len(st.session_state.buffer_leituras))
    
    col_fim1, col_fim2 = st.columns(2)
    if col_fim1.button(f"💾 SALVAR MÉDIA FINAL ({media_final})", type="primary", use_container_width=True):
        if not equipamento_atual:
            st.error("Precisa do nome do equipamento.")
        else:
            salvar_leitura(equipamento_atual, media_final, f"Média (N={len(st.session_state.buffer_leituras)})")
            st.balloons()
            st.session_state.buffer_leituras = []
            st.session_state.uploader_key += 1
            st.rerun()
            
    if col_fim2.button("❌ Limpar Memória", use_container_width=True):
        st.session_state.buffer_leituras = []
        st.rerun()