# recalque-app == app.py   
from utils import*


BASE_DIR = os.path.join( os.path.abspath('..') )
MODELS_DIR = os.path.join( BASE_DIR, 'models' )
IMGS_DIR = os.path.join( BASE_DIR, 'imgs' )
DATA_DIR = os.path.join(BASE_DIR, 'data')

pd.set_option('max_rows', 600)
pd.set_option('precision', 1)
def main():
    
    #! arquivo csv inicial para geração do modelo
    input_file = 'recalque_to_modeling.csv'
    input_path = os.path.join(DATA_DIR, input_file)
    df = pd.read_csv(input_path, index_col=0)
    
    page = st.sidebar.selectbox("Escolha uma página",['Homepage', 'Exploração', 'Predição'])
    
    if page == 'Homepage':     
        st.title('Exploração descritiva e aplicação de modelo de regressão para previsão do Recalque')
        st.text('Selecione uma página ao lado')
        st.dataframe(df.head())
        
    elif page == 'Exploração':
        st.title('Explore o dataset inicial')
        
        st.markdown('Número de amostras presentes:')
        st.markdown(df.shape[0])
        st.markdown('Número de variáveis presentes:')
        st.markdown(df.shape[1])
        
        if st.checkbox('Descrição estatística das colunas'):    
            st.dataframe(df.describe())
            
            
        st.markdown('### Analisando a distribuição do recalque')
        fig, (ax_hist, ax_box) = plt.subplots(2, sharex=True, 
        gridspec_kw={"height_ratios": (0.75, 0.25)})
        sns.histplot(
        data = df['s (mm)'],
        stat = 'count',
        bins = "auto",
        color = 'b',
        edgecolor = 'k',
        ax = ax_hist)
        sns.boxplot(x = df['s (mm)'], ax = ax_box, color = 'green')
        plt.xlim(0, 100)
        plt.title('Distribuição dos recalques até 100mm')
        st.pyplot(fig)
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('### Correlações das variáveis independentes com a variável target')
        st.markdown('<br/>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(),annot=True, fmt="2.2f", vmin=-1, vmax=1, center=0,
        cmap = 'coolwarm', ax = ax)
        st.pyplot(fig)

    else:     
        st.title("Modelagem: Regressão ")
        modelo, training_score, testing_score, rmse, df_results, y_test, preds= train_model(df)
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('Resultados das métricas da modelagem:')
        st.markdown('<br/>', unsafe_allow_html=True)
        st.write(f"Train R2 = {training_score:.3f}, Test R2 = {testing_score:.3f}")
        st.write(f"RMSE = {rmse:.2f}")
        st.dataframe(df_results.head())
        
        reais = df_results['s(mm)_real']
        previstos = df_results["s(mm)_estimado"]
        
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown("**Scatterplot do pontos de validação x predições**")
        fig, ax = plt.subplots()
        plt.scatter(y_test, preds, alpha = 0.5, color = 'b')
        plt.plot(y_test, y_test, color = 'red')
        st.pyplot(fig)
        
        #! Utilizando o modelo salvo em disco 
        
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('## Faça novas predições usando o modelo criado:')
        st.markdown("***")
        st.markdown('<br/>', unsafe_allow_html=True)
        

if __name__ == '__main__':
    main()
    