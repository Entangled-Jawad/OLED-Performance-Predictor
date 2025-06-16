import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="OLED Performance Predictor",
    page_icon="💡",
    layout="wide"
)

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        # Read the CSV file
        data = pd.read_csv('Blend_updated_csv.csv', skipinitialspace=True)
        
        # Clean column names - remove extra spaces, quotes, etc.
        data.columns = [col.strip() for col in data.columns]
        
        # Convert ratio columns to numeric, removing any text
        data['Material_A_ratio'] = pd.to_numeric(data['Material_A_ratio'], errors='coerce')
        data['Material_B_ratio'] = pd.to_numeric(data['Material_B_ratio'], errors='coerce')
        
        # Fill missing ratios with default values
        data['Material_A_ratio'].fillna(1.0, inplace=True)
        data['Material_B_ratio'].fillna(1.0, inplace=True)
        
        # Clean materials columns
        data['Material_A'] = data['Material_A'].str.strip()
        data['Material_B'] = data['Material_B'].str.strip()
        
        # Convert performance metrics to numeric
        numeric_cols = ['HOMO (eV)', 'LUMO (eV)', 'Bandgap (eV)', 'Thickness (nm)',
                    'EQE (%)', 'Lifetime (hrs)', 'Current Efficiency (cd/A)',
                    'Power Efficiency (lm/W)', 'Turn-on Voltage (V)']
        
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with missing values
        data = data.dropna()
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to train models and save them
def train_and_save_models(df):
    # Show data info before training
    st.write("Data shape before training:", df.shape)
    
    # Check if the required columns exist
    required_cols = ['Material_A', 'Material_B', 'HOMO (eV)', 'LUMO (eV)', 
                     'Bandgap (eV)', 'Thickness (nm)', 'EQE (%)', 'Lifetime (hrs)', 
                     'Current Efficiency (cd/A)', 'Power Efficiency (lm/W)', 'Turn-on Voltage (V)']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None, None, None, None, None
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Encode categorical features
    label_encoders = {}
    for col in ['Material_A', 'Material_B']:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Define features and targets
    features = ['Material_A_encoded', 'Material_B_encoded', 'Material_A_ratio', 'Material_B_ratio', 
                'HOMO (eV)', 'LUMO (eV)', 'Bandgap (eV)', 'Thickness (nm)']
    
    targets = ['EQE (%)', 'Lifetime (hrs)', 'Current Efficiency (cd/A)', 
               'Power Efficiency (lm/W)', 'Turn-on Voltage (V)']
    
    # Train a model for each target variable
    models = {}
    scalers = {}
    
    try:
        for target in targets:
            st.write(f"Training model for {target}...")
            
            # Split data
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Create sanitized filename
            safe_target = target.replace(" ", "_").replace("(%)", "").replace("/", "_per_")
            
            # Save model and scaler
            model_path = os.path.join('models', f'{safe_target}_model.pkl')
            scaler_path = os.path.join('models', f'{safe_target}_scaler.pkl')
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            st.write(f"Saved model and scaler for {target}")
            
            models[target] = (model, rmse, r2)
            scalers[target] = scaler
        
        # Save label encoders
        encoders_path = os.path.join('models', 'label_encoders.pkl')
        joblib.dump(label_encoders, encoders_path)
        st.write("Saved label encoders")
        
        return models, scalers, label_encoders, features, targets
        
    except Exception as e:
        st.error(f"Error during model training and saving: {str(e)}")
        return None, None, None, None, None

# Function to load saved models
def load_models(targets):
    models = {}
    scalers = {}
    
    for target in targets:
        # Create sanitized filename same way as in train_and_save_models
        safe_target = target.replace(" ", "_").replace("(%)", "").replace("/", "_per_")
        model_path = os.path.join('models', f'{safe_target}_model.pkl')
        scaler_path = os.path.join('models', f'{safe_target}_scaler.pkl')
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                models[target] = joblib.load(model_path)
                scalers[target] = joblib.load(scaler_path)
            else:
                st.error(f"Model or scaler not found for {target}! Looking for: {model_path}")
                return None, None, None
        except Exception as e:
            st.error(f"Error loading model for {target}: {str(e)}")
            return None, None, None
    
    try:
        encoders_path = os.path.join('models', 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
        else:
            st.error("Label encoders not found!")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading label encoders: {str(e)}")
        return None, None, None
    
    return models, scalers, label_encoders

# Function to make predictions
def predict_performance(models, scalers, label_encoders, input_data, features, targets):
    # Prepare input data
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col in ['Material_A', 'Material_B']:
        try:
            input_df[col + '_encoded'] = label_encoders[col].transform([input_data[col]])
        except:
            st.error(f"Unknown material: {input_data[col]} in {col}. Please select from available materials.")
            return None
    
    # Extract features
    X = input_df[features]
    
    # Make predictions for each target
    predictions = {}
    for target in targets:
        # Scale features
        X_scaled = scalers[target].transform(X)
        
        # Predict
        pred = models[target].predict(X_scaled)[0]
        predictions[target] = pred
    
    return predictions

# Main app function
def main():    
    st.title("💡 OLED Performance Predictor")
    
    st.markdown("""
    This application predicts the performance of Organic Light-Emitting Diodes (OLEDs) based on material properties and device parameters.
    """)
    
    # Tabs for app navigation
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Train Models", "Predict Performance"])
    
    # Load data
    df = load_data()
    
    # Check if data loaded successfully
    if df.empty:
        st.error("Failed to load data. Please check your CSV file.")
        return
    
    # Display data information
    st.sidebar.info(f"Loaded data with shape: {df.shape}")
    
    # Define targets
    targets = ['EQE (%)', 'Lifetime (hrs)', 'Current Efficiency (cd/A)', 
            'Power Efficiency (lm/W)', 'Turn-on Voltage (V)']
    
    with tab1:
        st.header("Dataset Overview")
        st.dataframe(df.head())
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
        # Only show material combinations if both columns exist
        if 'Material_A' in df.columns and 'Material_B' in df.columns:
            st.subheader("Material Combinations")
            material_counts = df.groupby(['Material_A', 'Material_B']).size().reset_index(name='Count')
            st.dataframe(material_counts)
        
        # Visualizations - only if data columns exist
        if 'EQE (%)' in df.columns and 'Bandgap (eV)' in df.columns:
            st.subheader("Data Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Distribution of EQE (%)")
                fig, ax = plt.subplots()
                sns.histplot(df['EQE (%)'], kde=True, ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.write("Relationship between Bandgap and EQE")
                fig, ax = plt.subplots()
                sns.scatterplot(x='Bandgap (eV)', y='EQE (%)', data=df, ax=ax)
                st.pyplot(fig)
    
    with tab2:
        st.header("Train Models")
        st.info("Click the button below to train and save predictive models for all OLED performance metrics.")
        
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                models, scalers, label_encoders, features, _ = train_and_save_models(df)
                
                if models:  # Check if models were created successfully
                    st.success("Models trained and saved successfully!")
                    
                    # Display model performance
                    st.subheader("Model Performance")
                    performance_data = []
                    for target, (model, rmse, r2) in models.items():
                        performance_data.append({
                            "Target": target,
                            "RMSE": rmse,
                            "R²": r2
                        })
                    
                    st.dataframe(pd.DataFrame(performance_data))
                else:
                    st.error("Model training failed. Please check your data.")
    
    with tab3:
        st.header("Predict OLED Performance")
        st.markdown("Enter the material properties and device parameters to predict OLED performance.")
        
        # Check if models directory exists
        if not os.path.exists('models') or not os.path.exists('models/label_encoders.pkl'):
            st.warning("Models not found. Please train the models first.")
        else:
            try:
                # Load models
                models, scalers, label_encoders = load_models(targets)
                features = ['Material_A_encoded', 'Material_B_encoded', 'Material_A_ratio', 'Material_B_ratio', 
                           'HOMO (eV)', 'LUMO (eV)', 'Bandgap (eV)', 'Thickness (nm)']
                
                # Get unique material options
                material_a_options = sorted(df['Material_A'].unique())
                material_b_options = sorted(df['Material_B'].unique())
                
                # Create input form
                with st.form("prediction_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        material_a = st.selectbox("Material A", material_a_options)
                        homo = st.number_input("HOMO (eV)", value=-5.5, step=0.01, format="%.2f")
                        bandgap = st.number_input("Bandgap (eV)", value=1.5, step=0.01, format="%.2f")
                        material_a_ratio = st.number_input("Material A Ratio", value=1.0, step=0.1, format="%.1f")
                    
                    with col2:
                        material_b = st.selectbox("Material B", material_b_options)
                        lumo = st.number_input("LUMO (eV)", value=-4.0, step=0.01, format="%.2f")
                        thickness = st.number_input("Thickness (nm)", value=100.0, step=5.0, format="%.1f")
                        material_b_ratio = st.number_input("Material B Ratio", value=1.0, step=0.1, format="%.1f")
                    
                    submit_button = st.form_submit_button("Predict")
                
                if submit_button:
                    # Prepare input data
                    input_data = {
                        'Material_A': material_a,
                        'Material_B': material_b,
                        'Material_A_ratio': material_a_ratio,
                        'Material_B_ratio': material_b_ratio,
                        'HOMO (eV)': homo,
                        'LUMO (eV)': lumo,
                        'Bandgap (eV)': bandgap,
                        'Thickness (nm)': thickness
                    }
                    
                    # Make predictions
                    predictions = predict_performance(models, scalers, label_encoders, input_data, features, targets)
                    
                    if predictions:
                        st.subheader("Predicted OLED Performance")
                        
                        # Display predictions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create gauge chart for EQE
                            fig, ax = plt.subplots(figsize=(5, 3))
                            eqe_pred = predictions['EQE (%)']
                            ax.barh(["EQE (%)"], [eqe_pred], color='skyblue')
                            ax.barh(["EQE (%)"], [100-eqe_pred], left=[eqe_pred], color='lightgray')
                            ax.set_xlim(0, 100)
                            plt.title(f"External Quantum Efficiency: {eqe_pred:.2f}%")
                            st.pyplot(fig)
                            
                            # Add lifetime prediction
                            st.metric("Lifetime (hours)", f"{predictions['Lifetime (hrs)']:.0f}")
                        
                        with col2:
                            # Show other metrics
                            st.metric("Turn-on Voltage (V)", f"{predictions['Turn-on Voltage (V)']:.2f}")
                            st.metric("Current Efficiency (cd/A)", f"{predictions['Current Efficiency (cd/A)']:.2f}")
                            st.metric("Power Efficiency (lm/W)", f"{predictions['Power Efficiency (lm/W)']:.2f}")
                        
                        # Summary table
                        st.subheader("Summary of Predictions")
                        summary_df = pd.DataFrame({
                            "Metric": list(predictions.keys()),
                            "Predicted Value": list(predictions.values())
                        })
                        st.dataframe(summary_df)
                        
                        # Recommendation
                        if predictions['EQE (%)'] > 20:
                            st.success("This combination has excellent EQE performance!")
                        elif predictions['EQE (%)'] > 10:
                            st.info("This combination has good EQE performance.")
                        else:
                            st.warning("This combination has below-average EQE performance.")
                            
            except Exception as e:
                st.error(f"Error loading models. Please train the models first. Error: {e}")
                st.info("Go to the 'Train Models' tab and click 'Train Models' button.")

if __name__ == "__main__":
    main()