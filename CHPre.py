import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from joblib import load
import warnings
import dice_ml
from dice_ml.utils import helpers

# 抑制FutureWarning（反事实逻辑需要）
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(2025)  # 确保可重复性

@st.cache_resource
def load_model():
    try:
        # Load your trained model (update path as needed)
        model = load('RF.pkl')  # Replace with your model path
        return model
    except FileNotFoundError:
        st.error("Model file not found! Ensure 'RF.pkl' is in the correct path.")
        return None

# Optimal threshold (from your analysis)
OPTIMAL_THRESHOLD = 0.27

# Feature definitions (English display names)
FEATURE_DETAILS = {
    'HospitalGrade': {'display': 'Hospital Grade', 'type': 'select', 'options': [1, 2, 3]},
    'Age': {'display': 'Age (years)', 'type': 'slider', 'min': 18, 'max': 90, 'default': 55},
    'Sex': {'display': 'Sex', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Female', 2: 'Male'}},
    'BMI': {'display': 'BMI', 'type': 'slider', 'min': 15, 'max': 40, 'default': 28},
    'InpatientStatus': {'display': 'Inpatient Status', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Outpatient', 2: 'Inpatient'}},
    'PreviousColonoscopy': {'display': 'Previous Colonoscopy', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'ChronicConstipation': {'display': 'Chronic Constipation', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DiabetesMellitus': {'display': 'Diabetes Mellitus', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'StoolForm': {'display': 'Stool Form', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Normal', 2: 'Abnormal'}},
    'BowelMovements': {'display': 'Bowel Movements (1-5)', 'type': 'select', 'options': [1, 2, 3, 4, 5]},
    'BPEducationModality': {'display': 'BP Education Modality', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Traditional', 2: 'Enhanced'}},
    'DietaryRestrictionDays': {'display': 'Dietary Restriction Days', 'type': 'slider', 'min': 0, 'max': 3, 'default': 1},
    'PreColonoscopyPhysicalActivity': {'display': 'Pre-Colonoscopy Physical Activity', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'PreviousAbdominopelvicSurgery_1.0': {'display': 'Previous Surgery Type 1', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'PreviousAbdominopelvicSurgery_2.0': {'display': 'Previous Surgery Type 2', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'PreviousAbdominopelvicSurgery_3.0': {'display': 'Previous Surgery Type 3', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_1': {'display': 'Diet Restriction Type 1', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_2': {'display': 'Diet Restriction Type 2', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_3': {'display': 'Diet Restriction Type 3', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_4': {'display': 'Diet Restriction Type 4', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_1': {'display': 'Laxative Regimen 1', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_2': {'display': 'Laxative Regimen 2', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_3': {'display': 'Laxative Regimen 3', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_4': {'display': 'Laxative Regimen 4', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_5': {'display': 'Laxative Regimen 5', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_6': {'display': 'Laxative Regimen 6', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}}
}

# Required feature order (match training data)
FEATURE_ORDER = list(FEATURE_DETAILS.keys())

def create_input_form():
    with st.form("patient_input_form"):
        st.subheader("Patient Feature Input")
        input_data = {}
        
        # Split into columns for better layout
        cols_main = st.columns(2)
        
        # Column 1: Basic demographics & clinical history
        with cols_main[0]:
            st.markdown("**Basic Information**")
            input_data['HospitalGrade'] = st.selectbox(FEATURE_DETAILS['HospitalGrade']['display'], FEATURE_DETAILS['HospitalGrade']['options'])
            input_data['Age'] = st.slider(FEATURE_DETAILS['Age']['display'], FEATURE_DETAILS['Age']['min'], FEATURE_DETAILS['Age']['max'], FEATURE_DETAILS['Age']['default'])
            input_data['Sex'] = st.selectbox(FEATURE_DETAILS['Sex']['display'], FEATURE_DETAILS['Sex']['options'], format_func=lambda x: FEATURE_DETAILS['Sex']['labels'][x])
            input_data['BMI'] = st.slider(FEATURE_DETAILS['BMI']['display'], FEATURE_DETAILS['BMI']['min'], FEATURE_DETAILS['BMI']['max'], FEATURE_DETAILS['BMI']['default'])
            
            st.markdown("**Clinical History**")
            input_data['InpatientStatus'] = st.selectbox(FEATURE_DETAILS['InpatientStatus']['display'], FEATURE_DETAILS['InpatientStatus']['options'], format_func=lambda x: FEATURE_DETAILS['InpatientStatus']['labels'][x])
            input_data['PreviousColonoscopy'] = st.selectbox(FEATURE_DETAILS['PreviousColonoscopy']['display'], FEATURE_DETAILS['PreviousColonoscopy']['options'], format_func=lambda x: FEATURE_DETAILS['PreviousColonoscopy']['labels'][x])
            input_data['ChronicConstipation'] = st.selectbox(FEATURE_DETAILS['ChronicConstipation']['display'], FEATURE_DETAILS['ChronicConstipation']['options'], format_func=lambda x: FEATURE_DETAILS['ChronicConstipation']['labels'][x])
            input_data['DiabetesMellitus'] = st.selectbox(FEATURE_DETAILS['DiabetesMellitus']['display'], FEATURE_DETAILS['DiabetesMellitus']['options'], format_func=lambda x: FEATURE_DETAILS['DiabetesMellitus']['labels'][x])
            
            st.markdown("**Gastrointestinal Features**")
            input_data['StoolForm'] = st.selectbox(FEATURE_DETAILS['StoolForm']['display'], FEATURE_DETAILS['StoolForm']['options'], format_func=lambda x: FEATURE_DETAILS['StoolForm']['labels'][x])
            input_data['BowelMovements'] = st.selectbox(FEATURE_DETAILS['BowelMovements']['display'], FEATURE_DETAILS['BowelMovements']['options'])

        # Column 2: Preparation & Surgery History
        with cols_main[1]:
            st.markdown("**Bowel Preparation**")
            input_data['BPEducationModality'] = st.selectbox(FEATURE_DETAILS['BPEducationModality']['display'], FEATURE_DETAILS['BPEducationModality']['options'], format_func=lambda x: FEATURE_DETAILS['BPEducationModality']['labels'][x])
            input_data['DietaryRestrictionDays'] = st.slider(FEATURE_DETAILS['DietaryRestrictionDays']['display'], FEATURE_DETAILS['DietaryRestrictionDays']['min'], FEATURE_DETAILS['DietaryRestrictionDays']['max'], FEATURE_DETAILS['DietaryRestrictionDays']['default'])
            input_data['PreColonoscopyPhysicalActivity'] = st.selectbox(FEATURE_DETAILS['PreColonoscopyPhysicalActivity']['display'], FEATURE_DETAILS['PreColonoscopyPhysicalActivity']['options'], format_func=lambda x: FEATURE_DETAILS['PreColonoscopyPhysicalActivity']['labels'][x])
            
            st.markdown("**Previous Surgery History**")
            input_data['PreviousAbdominopelvicSurgery_1.0'] = st.selectbox(FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['display'], FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['options'], format_func=lambda x: FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['labels'][x])
            input_data['PreviousAbdominopelvicSurgery_2.0'] = st.selectbox(FEATURE_DETAILS['PreviousAbdominopelvicSurgery_2.0']['display'], FEATURE_DETAILS['PreviousAbdominopelvicSurgery_2.0']['options'], format_func=lambda x: FEATURE_DETAILS['PreviousAbdominopelvicSurgery_2.0']['labels'][x])
            input_data['PreviousAbdominopelvicSurgery_3.0'] = st.selectbox(FEATURE_DETAILS['PreviousAbdominopelvicSurgery_3.0']['display'], FEATURE_DETAILS['PreviousAbdominopelvicSurgery_3.0']['options'], format_func=lambda x: FEATURE_DETAILS['PreviousAbdominopelvicSurgery_3.0']['labels'][x])
            
            st.markdown("**Diet Restriction Type**")
            cols_diet = st.columns(2)
            with cols_diet[0]:
                input_data['DietaryRestriction_1'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_1']['display'], FEATURE_DETAILS['DietaryRestriction_1']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_1']['labels'][x])
                input_data['DietaryRestriction_2'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_2']['display'], FEATURE_DETAILS['DietaryRestriction_2']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_2']['labels'][x])
            with cols_diet[1]:
                input_data['DietaryRestriction_3'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_3']['display'], FEATURE_DETAILS['DietaryRestriction_3']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_3']['labels'][x])
                input_data['DietaryRestriction_4'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_4']['display'], FEATURE_DETAILS['DietaryRestriction_4']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_4']['labels'][x])
            
            st.markdown("**Laxative Regimen**")
            cols_lax = st.columns(3)
            with cols_lax[0]:
                input_data['LaxativeRegimen_1'] = st.selectbox(FEATURE_DETAILS['LaxativeRegimen_1']['display'], FEATURE_DETAILS['LaxativeRegimen_1']['options'], format_func=lambda x: FEATURE_DETAILS['LaxativeRegimen_1']['labels'][x])
                input_data['LaxativeRegimen_2'] = st.selectbox(FEATURE_DETAILS['LaxativeRegimen_2']['display'], FEATURE_DETAILS['LaxativeRegimen_2']['options'], format_func=lambda x: FEATURE_DETAILS['LaxativeRegimen_2']['labels'][x])
            with cols_lax[1]:
                input_data['LaxativeRegimen_3'] = st.selectbox(FEATURE_DETAILS['LaxativeRegimen_3']['display'], FEATURE_DETAILS['LaxativeRegimen_3']['options'], format_func=lambda x: FEATURE_DETAILS['LaxativeRegimen_3']['labels'][x])
                input_data['LaxativeRegimen_4'] = st.selectbox(FEATURE_DETAILS['LaxativeRegimen_4']['display'], FEATURE_DETAILS['LaxativeRegimen_4']['options'], format_func=lambda x: FEATURE_DETAILS['LaxativeRegimen_4']['labels'][x])
            with cols_lax[2]:
                input_data['LaxativeRegimen_5'] = st.selectbox(FEATURE_DETAILS['LaxativeRegimen_5']['display'], FEATURE_DETAILS['LaxativeRegimen_5']['options'], format_func=lambda x: FEATURE_DETAILS['LaxativeRegimen_5']['labels'][x])
                input_data['LaxativeRegimen_6'] = st.selectbox(FEATURE_DETAILS['LaxativeRegimen_6']['display'], FEATURE_DETAILS['LaxativeRegimen_6']['options'], format_func=lambda x: FEATURE_DETAILS['LaxativeRegimen_6']['labels'][x])
        
        # Submit button
        submitted = st.form_submit_button("Predict")
        if submitted:
            # Reorder input data to match training order
            patient_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
            return patient_df
    return None

# ========== Model Wrapper for Threshold ==========
class ModelWrapper:
    def __init__(self, model, threshold=OPTIMAL_THRESHOLD):
        self.model = model
        self.threshold = threshold
    
    def predict(self, X):
        probs = self.model.predict_proba(X)
        return (probs[:, 1] > self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# ========== Counterfactual Generation ==========
# ========== Counterfactual Generation ==========
def generate_counterfactuals(model, patient_data):
    # Create dummy training data (optimized for more realistic distribution)
    n_samples = 200  # 增加样本数，让数据更接近真实分布
    dummy_train = {}
    for col in FEATURE_ORDER:
        feat_detail = FEATURE_DETAILS[col]
        if feat_detail['type'] == 'select':
            dummy_train[col] = np.random.choice(feat_detail['options'], size=n_samples)
        elif feat_detail['type'] == 'slider':
            if col == 'Age':
                dummy_train[col] = np.random.randint(20, 80, size=n_samples)  # 更合理的年龄范围
            elif col == 'DietaryRestrictionDays':
                dummy_train[col] = np.random.randint(0, 4, size=n_samples)
            elif col == 'BMI':
                dummy_train[col] = np.random.uniform(18, 35, size=n_samples)  # 更合理的BMI范围
    
    dummy_train = pd.DataFrame(dummy_train)
    dummy_train['outcome'] = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])  # 调整正负样本比例
    
    # Define feature types
    continuous_vars = ['Age', 'BMI', 'DietaryRestrictionDays']
    categorical_vars = [col for col in FEATURE_ORDER if col not in continuous_vars]
    
    # Create DiCE data object
    data = dice_ml.Data(
        dataframe=dummy_train,
        continuous_features=continuous_vars,
        categorical_features=categorical_vars,
        outcome_name='outcome'
    )
    
    # Wrap model
    wrapped_model = ModelWrapper(model)
    
    # Create DiCE model and explainer
    dice_model = dice_ml.Model(model=wrapped_model, backend='sklearn')
    exp = dice_ml.Dice(data, dice_model)
    
    # Relaxed constraints (easier to generate valid counterfactuals)
    def mutually_exclusive_constraint(instance):
        # 放宽约束：允许饮食限制/泻药方案有一个选中（非严格必须）
        diet_cols = [f'DietaryRestriction_{i}' for i in range(1, 5)]
        lax_cols = [f'LaxativeRegimen_{i}' for i in range(1, 7)]
        # 只要至少有一个选中即可，非严格恰好一个
        if sum(instance[col] for col in diet_cols) < 1 or sum(instance[col] for col in lax_cols) < 1:
            return False
        return True
    
    # Relaxed clinical check
    def clinical_plausibility_check(cf, original):
        issues = []
        # 注释年龄检查，或放宽BMI阈值
        # if cf['Age'] < original['Age'].iloc[0]:
        #     issues.append("Age cannot decrease")
        if abs(cf['BMI'] - original['BMI'].iloc[0]) > 10:  # 从5放宽到10
            issues.append(f"BMI change too large ({abs(cf['BMI'] - original['BMI'].iloc[0]):.1f})")
        return issues
    
    # Generate more counterfactual candidates
    try:
        dice_exp = exp.generate_counterfactuals(
            patient_data,
            total_CFs=20,  # 增加候选数
            features_to_vary=[col for col in FEATURE_ORDER if col != 'Age'],  # 只固定年龄，其他都可修改
            desired_class="opposite"    
        )
        
        # Filter valid counterfactuals
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        valid_cfs = []
        explanations = []
        
        for _, cf in cf_df.iterrows():
            if mutually_exclusive_constraint(cf):
                issues = clinical_plausibility_check(cf, patient_data)
                if not issues:
                    valid_cfs.append(cf)
                    # Generate explanation
                    changes = {}
                    for col in FEATURE_ORDER:
                        if cf[col] != patient_data[col].iloc[0]:
                            changes[FEATURE_DETAILS[col]['display']] = f"{patient_data[col].iloc[0]} → {cf[col]}"
                    explanations.append(changes)
        
        return pd.DataFrame(valid_cfs), explanations
    except Exception as e:
        st.error(f"Failed to generate counterfactuals: {str(e)}")
        import traceback
        st.code(traceback.format_exc())  # 显示详细错误，方便排查
        return pd.DataFrame(), []

# ========== SHAP Explanations ==========
def explain_prediction(model, patient_data):
    # Individual SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    
    # 提取单样本的SHAP值和基准值（分类模型取正类）
    base_value = explainer.expected_value[1]
    sample_shap = shap_values[1][0]  # 当前样本的SHAP值
    sample_features = patient_data.iloc[0]  # 当前样本的特征值
    feature_names = [FEATURE_DETAILS[col]['display'] for col in FEATURE_ORDER]
    
    # 创建Waterfall图（你想要的第二张图类型）
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sample_shap,
            base_values=base_value,
            data=sample_features,
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    return fig, shap_values[1]

# ========== Main Function ==========
def main():
    st.title("Bowel Preparation Outcome Predictor")
    st.markdown("""
    This tool predicts the probability of inadequate bowel preparation using a machine learning model,
    and provides counterfactual recommendations to improve outcomes.
    Enter patient features and click "Predict" to get results.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create input form
    patient_data = create_input_form()
    
    if patient_data is not None:
        # Display input data
        st.subheader("Patient Input Summary")
        display_df = patient_data.copy()
        display_df.columns = [FEATURE_DETAILS[col]['display'] for col in display_df.columns]
        st.dataframe(display_df.T, column_config={"0": "Value"}, use_container_width=True)
        
        # Predict
        wrapped_model = ModelWrapper(model)
        prob = wrapped_model.predict_proba(patient_data)[:, 1][0]
        prediction = 1 if prob > OPTIMAL_THRESHOLD else 0
        
        # Display prediction results
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Probability of Inadequate Preparation",
                f"{prob:.2%}",
                delta="High Risk" if prediction == 1 else "Low Risk",
                delta_color="inverse" if prediction == 1 else "normal"
            )
        with col2:
            st.write(f"Prediction: {'Inadequate' if prediction == 1 else 'Adequate'}")
        
        # SHAP Explanations
        st.subheader("Model Explanations")
        cols_shap = st.columns(2)
        
        # Global SHAP plot (pre-saved image)
        with cols_shap[0]:
            st.markdown("**Global Feature Importance (SHAP)**")
            try:
                st.image("global_shap_plot.png")  # Replace with your image path
            except FileNotFoundError:
                st.warning("Global SHAP plot not found (save as 'global_shap_plot.png')")
        
        # Individual SHAP plot
        with cols_shap[1]:
            st.markdown("**Individual Prediction Explanation**")
            shap_fig, shap_vals = explain_prediction(model, patient_data)
            st.pyplot(shap_fig)
        
        if prediction == 1:
            st.subheader("Counterfactual Recommendations")
            with st.spinner("Generating personalized recommendations..."):
                cf_df, explanations = generate_counterfactuals(model, patient_data)
            
            if not cf_df.empty:
                st.write("Valid Counterfactual Recommendations:")
                for i, (_, cf) in enumerate(cf_df.iterrows()):
                    st.markdown(f"**反事实 {i+1}**")
                    # 遍历当前反事实的特征变化
                    for feature, change in explanations[i].items():
                        st.write(f"- {feature}: {change}")
                    # 定义cf_display并展示
                    cf_display = cf[FEATURE_ORDER]  # 直接取反事实的所有特征列
                    cf_display.index = [FEATURE_DETAILS[col]['display'] for col in cf_display.index]
                    st.dataframe(cf_display.T, use_container_width=True)
            else:
                st.info("No valid counterfactual recommendations found.")

if __name__ == "__main__":
    main()