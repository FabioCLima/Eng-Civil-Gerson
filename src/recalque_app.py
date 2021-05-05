import os   
from utils import*
import streamlit as st 
import joblib
import pickle5 as pickle
import base64

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
        st.markdown('### Visualização das relações das variáveis presentes no dataset')
        st.markdown('<br/>', unsafe_allow_html=True)
        fig = sns.pairplot(df, diag_kind = 'kde', palette = 'Dark2', height = 1.5)
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
        
        st.markdown("**Scatterplot dos recalques: reais x predições**")
      
        
        fig = go.Figure(data=go.Scatter(
                        x = reais,
                        y = previstos,
                        mode = 'markers',
                        marker_color = reais,
                        text = df_results['s(mm)_real'])) 
        fig.add_trace(go.Scatter(
                        x = reais, 
                        y = reais,
                        mode='lines',
                        name='lines'))                                
        fig.update_layout(title='Scatterplot dos recalques - (reais x estimados)')
        st.plotly_chart(fig) 
        
        #! Utilizando o modelo salvo em disco 
        
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('## Faça novas predições usando o modelo criado:')
        st.markdown("***")
        st.markdown('<br/>', unsafe_allow_html=True)
        
        if os.path.isfile( os.path.join( MODELS_DIR, 'gbr_model.pkl' ) ):
            with open(os.path.join( MODELS_DIR, 'gbr_model.pkl' ), 'rb') as file_model:
                model = pickle.load(file_model)
        
        uploaded_file = st.file_uploader('Escolha o arquivo excel para usar o modelo do recalque (.xlsx)', type = 'xlsx')
        if uploaded_file is not None:
            
            def load_new_data(filename): 
                
                colunas = ['Caso', 's (mm)', 'L (m)', 'B (m)', 'Df (m)', 'q (kPa)', 'Nspt', 
                'ANJOS']
            
                new_data = pd.read_excel(filename,
                           sheet_name = 'Planilha1',
                           header = 1,
                           usecols = colunas,
                           index_col = 0,
                           engine = 'openpyxl')
                df = new_data.copy()
                return df

            df = load_new_data(filename = uploaded_file)
            
            new_data = df.drop(['s (mm)'], axis = 1)
            y_validation = df['s (mm)']
            data_scaler = model['model'].named_steps.preprocessor.transform(new_data)
            preds = model['model'].named_steps.gbr.predict(data_scaler)
            
            residuals = y_validation - preds
            df_new_preds = pd.DataFrame({
   
            's(mm)_estimado'     : np.round(y_pred, 3),    # recalque_estimado
            's(mm)_real'         : np.round(y_test, 3),    # recalque_real
            'erro_residual(mm)'  : np.round(residuals, 3)  # diferença
            })
            
            novas_previsoes = df_new_preds.reset_index(drop=True)
            
            st.dataframe(novas_previsoes)
            
    

if __name__ == '__main__':
    main()
    