import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from fpdf import FPDF
import pandas as pd

# Função para gerar PDF
def gerar_pdf(nome, cpf, idade, sexo, medicamentos, evolucao, info_medicas, recomendacoes):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='', size=12)

    pdf.cell(200, 10, "Relatório Médico", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Dados do Paciente:", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Nome: {nome}", ln=True)
    pdf.cell(200, 10, f"CPF: {cpf}", ln=True)
    pdf.cell(200, 10, f"Idade: {idade} anos", ln=True)
    pdf.cell(200, 10, f"Sexo: {sexo}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Medicamentos:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, medicamentos)
    pdf.ln(5)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Evoluções do tratamento:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, evolucao)
    pdf.ln(5)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Informações Médicas:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, info_medicas)
    pdf.ln(5)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, "Recomendações:", ln=True)
    pdf.set_font("Arial", size=12)
    for rec in recomendacoes:
        pdf.multi_cell(0, 10, f"- {rec}")
        pdf.ln(2)

    return pdf.output(dest='S').encode('latin1')

# Carregar modelo
MODEL_NAME = "NullisTerminis/ChemoToxAI"
label_map = {'Monitorar função cardíaca': 0, 'Avaliar sintomas respiratórios': 1, 'Ajustar hidratação e monitorar função renal': 2, 'Continuar protocolo atual': 3, 'Considerar profilaxia antiviral': 4, 'Ajuste de antihipertensivos': 5, 'Acompanhar função pulmonar': 6, 'Reduzir dose e monitorar intestino': 7, 'Manter controle de edema periférico': 8, 'Ajuste hormonal e monitoramento': 9, 'Controle da pressão arterial': 10, 'Ajustar doses com base na função cardíaca': 11, 'Acompanhamento nutricional': 12, 'Reforço nutricional e seguimento': 13, 'Monitorar glicemia': 14, 'Ajuste da dieta para obesidade': 15, 'Monitorar reações alérgicas': 16, 'Monitorar função renal': 17, 'Monitoramento hepático regular': 18, 'Reforço na hidratação': 19, 'Controle cardiovascular regular': 20, 'Avaliar função pulmonar': 21, 'Monitorar sinais de neuropatia': 22, 'Acompanhar sinais de alergia': 23, 'Manter hidratação adequada': 24, 'Monitorar função hepática': 25, 'Ajuste de dose com acompanhamento gastrointestinal': 26, 'Antiêmese reforçada': 27, 'Acompanhamento respiratório': 28, 'Seguir protocolo atual': 29, 'Controle de peso': 30, 'Ajuste de hormônios e monitoramento': 31, 'Controle glicêmico': 32, 'Monitorar colesterol': 33, 'Hidratação reforçada': 34, 'Monitorar pressão arterial': 35, 'Suporte pulmonar': 36, 'Manter controle do tratamento': 37, 'Monitoramento gastrointestinal': 38, 'Monitoramento de sintomas': 39, 'Monitoramento hepático': 40, 'Acompanhamento de alergia': 41, 'Hidratação adicional': 42, 'Controle de função respiratória': 43, 'Monitoramento cardíaco': 44, 'Controle padrão': 45, 'Suporte nutricional adicional': 46, 'Acompanhamento dermatológico': 47, 'Suporte respiratório': 48, 'Monitorar função hepática e intestinal': 49, 'Controle glicêmico regular': 50, 'Monitorar neuropatia': 51, 'Controle hematológico': 52, 'Controle gastrointestinal': 53, 'Suporte antiemético': 54, 'Monitoramento da função tireoidiana': 55, 'Monitoramento de função renal': 56, 'Monitoramento de perfil lipídico': 57, 'Acompanhamento padrão': 58, 'Acompanhamento respiratório e alergia': 59, 'Suporte respiratório e antihistamínicos': 60, 'Hidratação e monitoramento renal': 61, 'Acompanhamento hepático': 62, 'Monitoramento da pressão arterial': 63, 'Monitoramento cardíaco e antiemético': 64, 'Suporte gastrointestinal': 65, 'Acompanhamento de edema': 66, 'Ajuste de suporte': 67, 'Acompanhamento dermatológico e pressão arterial': 68, 'Monitoramento de fadiga': 69, 'Monitoramento glicêmico': 70, 'Monitoramento glicêmico e suporte energético': 71, 'Suporte respiratório e controle cardiovascular': 72, 'Controle hepático e suporte gastrointestinal': 73, 'Suporte dermatológico': 74, 'Monitoramento tireoidiano e suporte energético': 75, 'Controle da função renal': 76, 'Monitoramento respiratório': 77, 'Controle hepático e dermatológico': 78, 'Suporte energético': 79, 'Suporte respiratório e anti-histamínico': 80, 'Monitoramento cardíaco e gastrointestinal': 81, 'Monitoramento hematológico': 82, 'Monitoramento renal e pressão arterial': 83, 'Acompanhamento respiratório e suporte energético': 84, 'Controle cardíaco': 85, 'Monitoramento intestinal': 86, 'Monitoramento renal e suporte antiemético': 87, 'Suporte energético e ajuste de dose': 88, 'Acompanhamento respiratório e suporte antiemético': 89, 'Controle tireoidiano': 90, 'Suporte antiemético e monitoramento cardíaco': 91, 'Controle hepático': 92, 'Monitoramento renal e da pressão arterial': 93, 'Monitoramento tireoidiano': 94, 'Ajuste de dieta e hidratação': 95, 'Monitoramento renal e antiemético': 96, 'Monitoramento hepático e antiemético': 97, 'Ajuste de dieta': 98, 'Monitoramento hepático e suporte nutricional': 99, 'Ajuste de dieta e suporte hídrico': 100, 'Monitoramento renal': 101, 'Monitoramento glicêmico e ajuste de dieta': 102, 'Monitoramento hepático e suporte antiemético': 103, 'Monitoramento cardíaco frequente': 104, 'Ajuste de suporte antiemético': 105, 'Ajuste de medicação para controle glicêmico': 106, 'Monitoramento de função hepática': 107, 'Controle de sintomas e suporte nutricional': 108, 'Suporte para sintomas de hipotireoidismo': 109, 'Monitoramento renal regular': 110, 'Ajuste de medicação para controle pressórico': 111, 'Monitoramento dermatológico': 112, 'Monitoramento oftalmológico': 113, 'Monitoramento hepático contínuo': 114, 'Suporte para controle respiratório': 115, 'Ajuste de suporte para controle da fadiga': 116, 'Monitoramento de pressão arterial': 117, 'Monitoramento renal frequente': 118, 'Suporte cardiovascular contínuo': 119, 'Suporte para fadiga contínuo': 120, 'Monitoramento de pressão arterial contínuo': 121, 'Suporte nutricional contínuo': 122, 'Suporte antipirético contínuo': 123, 'Monitoramento respiratório contínuo': 124, 'Monitoramento renal e ajuste de dieta': 125, 'Monitoramento cardíaco contínuo': 126, 'Monitoramento para controle da fadiga': 127, 'Suporte contínuo para controle de fadiga': 128, 'Monitoramento hepático e ajuste de suporte': 129, 'Ajuste de suporte respiratório': 130, 'Suporte alérgico contínuo': 131, 'Monitoramento de glicose': 132, 'Suporte contínuo para fadiga': 133, 'Monitoramento de glicose contínuo': 134, 'Ajuste de suporte hepático': 135, 'Monitoramento renal contínuo': 136, 'Suporte para controle de peso': 137, 'Suporte para controle glicêmico': 138, 'Monitoramento cardiovascular contínuo': 139, 'Suporte contínuo para controle da fadiga': 140, 'Suporte cardíaco contínuo': 141, 'Monitoramento glicêmico contínuo': 142, 'Suporte para controle de náusea': 143, 'Suporte respiratório contínuo': 144, 'Monitoramento contínuo de pressão arterial': 145, 'Suporte para controle de fadiga': 146, 'Suporte hepático contínuo': 147, 'Suporte para controle de diarreia': 148}
label_map_invertido = {v: k for k, v in label_map.items()}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Inicializar o estado de sessão
if 'pacientes' not in st.session_state:
    st.session_state.pacientes = [
        {"nome": "João Silva", "cpf": "123.456.789-00", "idade": 45, "sexo": "Masculino",
         "medicamentos": "Doxorrubicina 50 mg/m², Ciclofosfamida 500 mg/²",
         "evolucao": "S1: Náusea leve (Grau 1), fadiga (Grau 2); S2: Neutropenia (Grau 3); S3: Melhor resposta após redução de dose",
         "info_medicas": "O paciente encontra-se no 3º ciclo de quimioterapia, apresentando fadiga moderada e episódios ocasionais de náusea."},

        {"nome": "Maria Oliveira", "cpf": "987.654.321-00", "idade": 60, "sexo": "Feminino",
         "medicamentos": "Paclitaxel 175 mg/m², Carboplatina AUC 5",
         "evolucao": "S1: Reação alérgica leve (Grau 1); S2: Neuropatia periférica leve (Grau 2); S3: Sem novos sintomas",
         "info_medicas": "A paciente está no 2º ciclo de quimioterapia, apresentando neuropatia periférica leve, sem impacto significativo na funcionalidade."},

        {"nome": "Carlos Souza", "cpf": "456.789.123-00", "idade": 50, "sexo": "Masculino",
         "medicamentos": "Capecitabina 1000 mg/m² 2x/dia",
         "evolucao": "S1: Diarreia moderada (Grau 2); S2: Lesão oral leve (Grau 1); S3: Sintomas controlados com suporte nutricional",
         "info_medicas": "O paciente segue para o 4º ciclo de quimioterapia, com sintomas gastrointestinais leves, incluindo diarreia esporádica."}
    ]

if 'paciente_selecionado' not in st.session_state:
    st.session_state.paciente_selecionado = None

st.title("Recomendações Oncológicas")

col1, col2 = st.columns(2)

with col1:
    if st.button('Consultar pacientes'):
        st.session_state.mostrar_consulta = True

    if st.session_state.get('mostrar_consulta', False):
        df_pacientes = pd.DataFrame(st.session_state.pacientes)
        st.dataframe(df_pacientes)

        index_paciente = st.selectbox("Selecione o ID do paciente: ", list(df_pacientes.index))
        st.session_state.paciente_selecionado = st.session_state.pacientes[index_paciente]

        if st.session_state.paciente_selecionado:
            paciente = st.session_state.paciente_selecionado

            nome = paciente['nome']
            cpf = paciente['cpf']
            idade = paciente['idade']
            sexo = paciente['sexo']
            medicamentos = paciente['medicamentos']
            evolucao = paciente['evolucao']
            informacoes_medicas = paciente['info_medicas']

            if nome.strip() and informacoes_medicas.strip():
                input_text = f"Paciente: {nome}, Idade: {idade}, Sexo: {sexo}. Histórico: {informacoes_medicas}"
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=-1)

                top_3_indices = torch.topk(predictions, 3).indices[0].tolist()
                recomendacoes = [label_map_invertido.get(idx, "Desconhecida") for idx in top_3_indices]

                pdf_bytes = gerar_pdf(nome, cpf, idade, sexo, medicamentos, evolucao, informacoes_medicas, recomendacoes)
                st.download_button(label="Baixar PDF", data=pdf_bytes, file_name="relatorio_medico.pdf", mime="application/pdf")
            else:
                st.warning("Por favor, preencha todos os campos obrigatórios.")

with col2:
    if st.button("Acrescentar paciente"):
        st.session_state.mostrar_formulario = True

    if st.session_state.get('mostrar_formulario', False):
        nome = st.text_input("Nome do paciente:")
        cpf = st.text_input("CPF do paciente:")
        idade = st.number_input("Idade:", min_value=0, max_value=120, step=1)
        sexo = st.selectbox("Sexo:", ["Masculino", "Feminino", "Outro", "Prefiro não dizer"])
        medicamentos = st.text_area("Medicamentos utilizados:")
        evolucao = st.text_area("Evolução do tratamento:")
        informacoes_medicas = st.text_area("Informações Médicas:", height=150)

        if st.button("Salvar Paciente"):
            novo_paciente = {
                "nome": nome,
                "cpf": cpf,
                "idade": idade,
                "sexo": sexo,
                "medicamentos": medicamentos,
                "evolucao": evolucao,
                "info_medicas": informacoes_medicas
            }
            st.session_state.pacientes.append(novo_paciente)
            st.success("Paciente adicionado com sucesso!")
            st.session_state.mostrar_formulario = False