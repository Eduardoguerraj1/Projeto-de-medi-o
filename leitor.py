import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import os
import cv2
import pandas as pd
from datetime import datetime
import time
from fpdf import FPDF

# ==========================================
# CONFIGURAÇÃO VISUAL (VIBRANTE & MODERNA)
# ==========================================
st.set_page_config(page_title="Leitor V26 - Pro Report", page_icon="📊", layout="wide")

# CSS PERSONALIZADO
st.markdown("""
<style>
    /* Fundo geral e fontes */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Botões Grandes e Coloridos */
    div.stButton > button:first-child {
        height: 3.8em;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Hover effects */
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }

    /* Cards de Informação */
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 6px solid #2196F3;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.2);
    }
    
    /* Status Boxes (Aprovado/Reprovado) */
    .status-box {
        padding: 25px;
        text-align: center;
        border-radius: 20px;
        margin: 20px 0;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        animation: fadeIn 0.5s;
    }
    .status-ok {
        background: linear-gradient(45deg, #2e7d32, #66bb6a);
        border: 2px solid #fff;
    }
    .status-nok {
        background: linear-gradient(45deg, #c62828, #ef5350);
        border: 2px solid #fff;
    }
    
    /* Texto Grande */
    .big-text {
        font-size: 40px;
        font-weight: 900;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Badge de IA */
    .ai-badge {
        background: linear-gradient(90deg, #f8bbd0, #fce4ec);
        color: #880e4f;
        padding: 10px;
        border-radius: 8px;
        font-size: 14px;
        border: 1px solid #f48fb1;
        margin-bottom: 15px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* Animação */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 0. DADOS TÉCNICOS
# ==========================================
DADOS_TECNICOS = {
    "FH-40": {
        "Valor Padrão": "200 nSv/h",
        "Tolerância": "± 10%",
        "Faixa Aceitável": "180 - 220 nSv/h",
        "Nº de série": "FH-001-X",
        "Fonte Calibração": "Interna",
        "Carregador": "Bateria 9V"
    },
    "Fixo na maleta VacuTec 70043- A": {
        "Nº de série": "99002",
        "Ref. Esperada": "1,39 uSv/h (1,25 - 1,53)",
        "Erro aceitável": "± 10%",
        "Fonte Calibração": "Cs-137 CS-7A",
        "Carregador": "Universal 15 V DC - 6 A"
    },
    "Portátil VacuTec 70043- A": {
        "Nº de série": "00014",
        "Ref. Esperada": "2,88 uSv/h (2,59 - 3,17)",
        "Erro aceitável": "± 10%",
        "Fonte Calibração": "Cs-137 CS-7A (8 Ci)",
        "Carregador": "PUP55-13 Protek + adaptador"
    },
    "Portátil VacuTec 70046- A": {
        "Nº de série": "1800124",
        "Ref. Esperada": "3,07 uSv/h (2,76 - 3,38)",
        "Erro aceitável": "± 10%",
        "Fonte Calibração": "Cs-137 CS-7A",
        "Carregador": "PUP55-13 Protek + adaptador"
    }
}

ARQUIVO_CSV = "leituras_equipamentos.csv"

# ==========================================
# 1. FUNÇÕES DE PDF (RELATÓRIO)
# ==========================================
class PDFRelatorio(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Relatorio Tecnico - Controle Radiometrico', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def gerar_pdf_diario():
    if not os.path.exists(ARQUIVO_CSV):
        return None
    
    df = pd.read_csv(ARQUIVO_CSV)
    # Filtra apenas hoje
    hoje_str = datetime.now().strftime("%Y-%m-%d")
    df['Data_So_Dia'] = df['Data'].apply(lambda x: x.split(' ')[0])
    df_hoje = df[df['Data_So_Dia'] == hoje_str]
    
    if df_hoje.empty:
        return None

    pdf = PDFRelatorio()
    pdf.add_page()
    
    # 1. Tabela de Medições
    pdf.chapter_title(f'1. Medicoes do Dia ({len(df_hoje)} registros)')
    pdf.set_font('Courier', 'B', 10)
    pdf.cell(40, 7, 'Hora', 1)
    pdf.cell(80, 7, 'Equipamento', 1)
    pdf.cell(30, 7, 'Leitura', 1)
    pdf.cell(30, 7, 'Tipo', 1)
    pdf.ln()
    
    pdf.set_font('Courier', '', 10)
    equipamentos_usados = set()
    
    for index, row in df_hoje.iterrows():
        hora = row['Data'].split(' ')[1]
        equip = row['Equipamento']
        val = str(row['Leitura'])
        tipo = row['Tipo']
        equipamentos_usados.add(equip)
        
        # Encurtar nome se for muito longo no PDF
        nome_display = (equip[:25] + '..') if len(equip) > 25 else equip
        
        pdf.cell(40, 7, hora, 1)
        pdf.cell(80, 7, nome_display, 1)
        pdf.cell(30, 7, val, 1)
        pdf.cell(30, 7, tipo, 1)
        pdf.ln()
    
    pdf.ln(10)
    
    # 2. Informações dos Equipamentos
    pdf.chapter_title('2. Detalhes dos Equipamentos Utilizados')
    
    for eq in equipamentos_usados:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"> {eq}", 0, 1)
        pdf.set_font('Arial', '', 10)
        
        if eq in DADOS_TECNICOS:
            dados = DADOS_TECNICOS[eq]
            texto_specs = ""
            for k, v in dados.items():
                texto_specs += f"   - {k}: {v}\n"
            pdf.multi_cell(0, 5, texto_specs)
        else:
            pdf.cell(0, 5, "   - Nenhuma especificacao cadastrada.", 0, 1)
        pdf.ln(3)

    # Output
    return pdf.output(dest='S').encode('latin-1', 'replace')

# ==========================================
# 2. ESTADO E CONFIGURAÇÃO
# ==========================================
if 'buffer_leituras' not in st.session_state: st.session_state.buffer_leituras = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0
if 'equipamento_selecionado' not in st.session_state: st.session_state.equipamento_selecionado = None
if 'modo_vacutec' not in st.session_state: st.session_state.modo_vacutec = False

def salvar_leitura(modelo_equip, valor, tipo="Única"):
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    novo_dado = pd.DataFrame([{"Data": data_hora, "Equipamento": modelo_equip, "Leitura": int(valor), "Tipo": tipo}])
    if not os.path.exists(ARQUIVO_CSV): novo_dado.to_csv(ARQUIVO_CSV, index=False)
    else: novo_dado.to_csv(ARQUIVO_CSV, mode='a', header=False, index=False)
    return True

# --- FUNÇÕES VISUAIS E DE DETECÇÃO (MANTIDAS) ---
def verificar_faixa_fh40(valor_lido):
    alvo = 200; tolerancia = 0.10
    minimo = int(alvo * (1 - tolerancia)); maximo = int(alvo * (1 + tolerancia))
    return (minimo <= valor_lido <= maximo), minimo, maximo

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
            merged.append((curr_x, curr_y, curr_w, curr_h)); curr_x, curr_y, curr_w, curr_h = next_x, next_y, next_w, next_h
    merged.append((curr_x, curr_y, curr_w, curr_h))
    return merged

def filtrar_outliers_horizontais(caixas, max_gap_x):
    if not caixas: return []
    caixas.sort(key=lambda c: c[0])
    grupos = []
    grupo_atual = [caixas[0]]
    for i in range(1, len(caixas)):
        box_anterior = caixas[i-1]; box_atual = caixas[i]
        if (box_atual[0] - (box_anterior[0] + box_anterior[2])) > max_gap_x: grupos.append(grupo_atual); grupo_atual = [box_atual]
        else: grupo_atual.append(box_atual)
    grupos.append(grupo_atual)
    return max(grupos, key=lambda g: sum(box[2] for box in g))

def separar_digitos_colados(caixas):
    novas_caixas = []
    for (x, y, w, h) in caixas:
        ratio = w / float(h)
        if ratio > 0.85: 
            metade_w = w // 2; novas_caixas.append((x, y, metade_w, h)); novas_caixas.append((x + metade_w, y, metade_w, h))
        else: novas_caixas.append((x, y, w, h))
    return novas_caixas

def segmentar_digitos(img_recortada_pil, params):
    img_cv = np.array(img_recortada_pil.convert('RGB'))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    if params.get('usar_clahe', True): gray = aplicar_clahe(gray, clip_limit=params['clahe_clip'])
    elif params['contraste_alto'] > 1.0: gray = cv2.convertScaleAbs(gray, alpha=params['contraste_alto'], beta=0)
    k = params['blur_k'] | 1 
    gray_blurred = cv2.GaussianBlur(gray, (k, k), 0)
    metodo = cv2.THRESH_BINARY_INV if params['inverter_bin'] else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray_blurred, params['thresh_val'], 255, metodo)
    if params.get('dilatacao_iter', 0) > 0:
        kernel = np.ones((3,3), np.uint8); thresh = cv2.dilate(thresh, kernel, iterations=params['dilatacao_iter'])
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
        max_h = max(c[3] for c in caixas_separadas); caixas_finais = [c for c in caixas_separadas if c[3] > (max_h * 0.40)]
    else: caixas_finais = []
    if len(caixas_finais) > 3: caixas_finais.sort(key=lambda c: c[0]); caixas_finais = caixas_finais[:3]
    img_debug = gray.copy(); img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2RGB)
    recortes = []; margin = params['margem']; h_img, w_img = gray.shape
    for (x, y, w, h) in caixas_finais:
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        y1, y2 = max(0, y-margin), min(h_img, y+h+margin); x1, x2 = max(0, x-margin), min(w_img, x+w+margin)
        recortes.append(Image.fromarray(img_cv[y1:y2, x1:x2]))
    return recortes, img_debug, thresh

def processar_para_yolo_digits(img_pil):
    img_gray = img_pil.convert("L")
    if np.mean(np.array(img_gray)) < 127: img_gray = ImageOps.invert(img_gray)
    img_rgb = img_gray.convert("RGB"); largura, altura = img_rgb.size; novo = max(largura, altura)
    img = Image.new("RGB", (novo, novo), (255, 255, 255)); img.paste(img_rgb, ((novo-largura)//2, (novo-altura)//2))
    return img

def detectar_e_recortar_tela(img_pil, model_det, conf, crop_fraction):
    results = model_det(img_pil, conf=conf)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0: return img_pil, False, None, 0.0
    box_data = results[0].boxes[0]; box = box_data.xyxy[0].cpu().numpy()
    class_id = int(box_data.cls[0].item()); class_name = results[0].names[class_id]; confidence = box_data.conf[0].item()
    x1, y1, x2, y2 = box; w, h = x2-x1, y2-y1; cx, cy = x1+w/2, y1+h/2; nw, nh = w*crop_fraction, h*crop_fraction
    crop = img_pil.crop((max(0, cx-nw/2), max(0, cy-nh/2), min(img_pil.width, cx+nw/2), min(img_pil.height, cy+nh/2)))
    return crop, True, class_name, confidence

# --- LOAD MODELS ---
PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
arquivos_pt = [f for f in os.listdir(PASTA_ATUAL) if f.endswith('.pt')]
st.sidebar.header("⚙️ Ajustes / Relatórios")

# --- BOTÃO DE DOWNLOAD PDF NO SIDEBAR ---
if st.sidebar.button("📄 Baixar Relatório PDF (Hoje)", type="primary"):
    pdf_bytes = gerar_pdf_diario()
    if pdf_bytes:
        st.sidebar.download_button(
            label="⬇️ Clique para Salvar PDF",
            data=pdf_bytes,
            file_name=f"Relatorio_Medicao_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    else:
        st.sidebar.error("Nenhuma medição encontrada hoje.")

usar_clahe = st.sidebar.checkbox("Anti-Reflexo", value=True)
clahe_force = st.sidebar.slider("Intensidade CLAHE", 1.0, 8.0, 3.0)

# Load Models
try:
    detector = YOLO(os.path.join(PASTA_ATUAL, "modelo_de_tela.pt"))
    classificador = YOLO(os.path.join(PASTA_ATUAL, "modelo_de_numero.pt"))
except: st.error("Erro carregando modelos."); st.stop()

params = {
    'contraste_alto': 1.5, 'usar_norm': False, 'inverter_bin': True,
    'blur_k': 3, 'thresh_val': 15, 'margem': 2, 'max_dist_y': 100, 'max_gap_x': 100, 
    'dilatacao_iter': 2, 'area_min_perc': 0.001, 'usar_clahe': usar_clahe, 'clahe_clip': clahe_force
}

# ==========================================
# 4. INTERFACE PRINCIPAL
# ==========================================
st.markdown("### 1. Seleção de Equipamento")

c1, c2, c3 = st.columns(3)
if c1.button("FH-40", use_container_width=True):
    st.session_state.equipamento_selecionado = "FH-40"; st.session_state.modo_vacutec = False; st.rerun()
if c2.button("VacuTec ▾", use_container_width=True):
    st.session_state.modo_vacutec = True; st.session_state.equipamento_selecionado = None; st.rerun()
if c3.button("Ludlum", use_container_width=True):
    st.session_state.equipamento_selecionado = "Ludlum"; st.session_state.modo_vacutec = False; st.rerun()

if st.session_state.modo_vacutec:
    st.info("📂 Selecione o modelo VacuTec:")
    s1, s2, s3 = st.columns(3)
    if s1.button("Fixo Maleta 70043-A", use_container_width=True): st.session_state.equipamento_selecionado = "Fixo na maleta VacuTec 70043- A"; st.rerun()
    if s2.button("Portátil 70043-A", use_container_width=True): st.session_state.equipamento_selecionado = "Portátil VacuTec 70043- A"; st.rerun()
    if s3.button("Portátil 70046-A", use_container_width=True): st.session_state.equipamento_selecionado = "Portátil VacuTec 70046- A"; st.rerun()

st.divider()

# EXIBE EQUIPAMENTO SELECIONADO
if st.session_state.equipamento_selecionado:
    eq = st.session_state.equipamento_selecionado
    # Banner colorido
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #4CAF50, #81c784); color:white; padding:15px; border-radius:10px; text-align:center; font-size:22px; font-weight:bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        ✅ EQUIPAMENTO: {eq}
    </div>
    """, unsafe_allow_html=True)
    
    if eq in DADOS_TECNICOS:
        dados = DADOS_TECNICOS[eq]
        linhas = "".join([f"<div class='info-row'><span style='font-weight:bold; color:#333;'>{k}:</span><span style='color:{'#d32f2f' if 'Ref' in k else '#1976D2'}; font-weight:bold;'>{v}</span></div>" for k, v in dados.items()])
        st.markdown(f"<div class='info-card'><div style='color:#0d47a1; font-weight:bold; font-size:18px; margin-bottom:10px;'>ℹ️ Especificações Técnicas</div>{linhas}</div>", unsafe_allow_html=True)
else:
    st.markdown("""<div style="background: linear-gradient(90deg, #ef5350, #e57373); color:white; padding:15px; border-radius:10px; text-align:center; font-weight:bold;">⚠️ AGUARDANDO SELEÇÃO (OU DETECÇÃO AUTO)</div>""", unsafe_allow_html=True)

# ==========================================
# 5. CAPTURA
# ==========================================
st.markdown("### 2. Captura de Imagem")
cam = st.camera_input("Câmera", label_visibility="collapsed")
upl = st.file_uploader("Upload", type=['jpg','png','jpeg'], key=f"up_{st.session_state.uploader_key}", label_visibility="collapsed")

img, origem = None, ""
if cam: img = Image.open(cam); origem = "camera"
elif upl: img = Image.open(upl); origem = "upload"

if img:
    st.markdown("---")
    # Detecção
    img_tela, achou, cls_name, conf = detectar_e_recortar_tela(img, detector, 0.15, 1.0)
    
    # Auto-Select FH-40
    if achou and not st.session_state.equipamento_selecionado and conf > 0.6:
        st.session_state.equipamento_selecionado = "FH-40"
        st.session_state.modo_vacutec = False
        st.toast(f"🤖 IA Identificou: {cls_name}. Selecionando FH-40...", icon="🧠")
        time.sleep(0.5); st.rerun()

    if not st.session_state.equipamento_selecionado:
        st.error("🛑 Selecione um equipamento acima para processar.")
    else:
        st.markdown(f"<div class='ai-badge'>🧠 IA ATIVA | Objeto: {cls_name} ({conf:.1%}) | Processamento Neural</div>", unsafe_allow_html=True)
        
        with st.spinner("🔍 Analisando dígitos..."):
            recortes, img_debug, thresh_img = segmentar_digitos(img_tela, params)
            leitura_str = ""
            for r in recortes:
                im = processar_para_yolo_digits(r)
                res = classificador(im, verbose=False)
                leitura_str += res[0].names[res[0].probs.top1]
            
            c1, c2, c3 = st.columns(3)
            c1.image(img_tela, caption="Recorte Tela")
            c2.image(img_debug, caption="Visão Robô")
            c3.image(thresh_img, caption="Binário")

            val = None
            try:
                limpo = "".join(filter(str.isdigit, leitura_str))
                if limpo:
                    val = int(limpo)
                    # VALIDAÇÃO FH-40 (PAINEL VERDE/VERMELHO)
                    if st.session_state.equipamento_selecionado == "FH-40":
                        ok, mn, mx = verificar_faixa_fh40(val)
                        cor, icone, txt = ("status-ok", "✅", "APROVADO") if ok else ("status-nok", "🚫", "REPROVADO")
                        st.markdown(f"""
                        <div class="status-box {cor}">
                            <div class="big-text">{icone} {txt}</div>
                            <div style="font-size:20px; margin-top:10px;">Leitura: {val} nSv/h</div>
                            <div style="background:rgba(0,0,0,0.2); display:inline-block; padding:5px 10px; border-radius:5px; margin-top:5px;">Meta: {mn} a {mx} nSv/h</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div style="text-align:center; margin:20px;"><span style="font-size:70px; font-weight:900; color:#1976D2;">{val}</span><br><span style="color:gray;">VALOR LIDO</span></div>""", unsafe_allow_html=True)
                else: st.warning("⚠️ Dígitos não reconhecidos.")
            except: st.error("Erro processamento.")

            if val is not None:
                b1, b2, b3 = st.columns(3)
                if b1.button("💾 SALVAR", type="primary", use_container_width=True):
                    salvar_leitura(st.session_state.equipamento_selecionado, val, "Única")
                    st.success("Registro salvo!"); st.session_state.uploader_key += 1 if origem=="upload" else 0; time.sleep(1); st.rerun()
                
                txt_add = f"➕ SOMAR ({len(st.session_state.buffer_leituras)})" if st.session_state.buffer_leituras else "➕ NA MÉDIA"
                if b2.button(txt_add, use_container_width=True):
                    st.session_state.buffer_leituras.append(val); st.session_state.uploader_key += 1 if origem=="upload" else 0; st.rerun()
                
                if b3.button("🗑️ DESCARTAR", use_container_width=True):
                    st.session_state.uploader_key += 1 if origem=="upload" else 0; st.rerun()

if st.session_state.buffer_leituras:
    st.markdown("---")
    med = int(sum(st.session_state.buffer_leituras)/len(st.session_state.buffer_leituras))
    if st.button(f"📊 SALVAR MÉDIA: {med}", type="primary", use_container_width=True):
        salvar_leitura(st.session_state.equipamento_selecionado, med, "Média")
        st.session_state.buffer_leituras = []; st.balloons(); st.rerun()
