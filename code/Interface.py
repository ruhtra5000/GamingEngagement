from sklearn.naive_bayes import GaussianNB
import streamlit as st

# Vector to convert prediction to text
answerVector = ["BAIXO", "MÉDIO", "ALTO"]

# Create web interface with streamlit
def showWebInterface(model: GaussianNB):
    # Setting content width
    st.set_page_config(layout="wide")
    columns = st.columns([0.2, 0.6, 0.2], vertical_alignment="center")

    # Title and form
    with columns[1]:
        st.title("Previsão de Engajamento em Jogos")
        
        with st.form("form"): 
            age = st.number_input("Idade", min_value=0)
            gameName = st.text_input("Nome do jogo")
            sessionsPerWeek = st.number_input("Sessões por semana", min_value=1)
            avgSessionLength = st.number_input("Tamanho médio das sessões (Minutos)", min_value=1)
            playerLevel = st.number_input("Nível no jogo", min_value=1)
            achievementsUnlocked = st.number_input("Qtde de conquistas desbloqueadas", min_value=0)

            submitted = st.form_submit_button("Calcular engajamento")
            
            if submitted:
                # Predicting engagement level
                prediction = model.predict([[age, sessionsPerWeek, avgSessionLength, 
                                        playerLevel, achievementsUnlocked]])[0]

                st.markdown(f"Seu nível de engajamento em {gameName} é :blue-background[{answerVector[prediction]}].")