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
    'HospitalGrade': {'display': 'Hospital Grade', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Non-Tertiary', 2: 'Tertiary'}, 'default': 2},
    'Age': {'display': 'Age (years)', 'type': 'slider', 'min': 18, 'max': 90, 'default': 55},
    'Sex': {'display': 'Sex', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Female', 2: 'Male'}},
    'BMI': {'display': 'BMI', 'type': 'slider', 'min': 15, 'max': 40, 'default': 28},
    'InpatientStatus': {'display': 'Inpatient Status', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Outpatient', 2: 'Inpatient'}},
    'PreviousColonoscopy': {'display': 'Previous Colonoscopy', 'type': 'select', 'options': [1, 2], 'labels': {2: 'No', 1: 'Yes'}},
    'ChronicConstipation': {'display': 'Chronic Constipation', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DiabetesMellitus': {'display': 'Diabetes Mellitus', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'StoolForm': {'display': 'Stool Form', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Bristol Stool Form Scale types 3-7', 2: 'Bristol Stool Form Scale types 1-2'}},
    'BowelMovements': {'display': 'Bowel Movements', 'type': 'select', 'options': [1, 2, 3, 4], 'labels': {1: '<5', 2: '5-10', 3: '10-20', 4: '≥20'}},
    'BPEducationModality': {'display': 'BP Education Modality', 'type': 'select', 'options': [1, 2], 'labels': {1: 'Enhanced (image or animation-based education)', 2: 'Traditional (text or verbal education)'},'default': 2},
    'DietaryRestrictionDays': {'display': 'Dietary Restriction Days', 'type': 'slider', 'min': 0.0, 'max': 3.0,  'step': 0.5, 'default':0.5},
    'PreColonoscopyPhysicalActivity': {'display': 'Pre-Colonoscopy Physical Activity', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'PreviousAbdominopelvicSurgery_1.0': {'display': 'History of abdominal/pelvic surgery', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'},'default': 1},
    'PreviousAbdominopelvicSurgery_2.0': {'display': 'History of abdominal surgery', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'PreviousAbdominopelvicSurgery_3.0': {'display': 'History of pelvic surgery', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_1': {'display': 'Fasting', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_2': {'display': 'Low-residue', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_3': {'display': 'Clear liquid', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'DietaryRestriction_4': {'display': 'Regular', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_1': {'display': 'PEG 2L', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_2': {'display': 'PEG 3L', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_3': {'display': 'PEG 4L', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_4': {'display': 'Sodium phosphate', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_5': {'display': 'Combination regimen', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}},
    'LaxativeRegimen_6': {'display': 'Magnesium salts', 'type': 'select', 'options': [0, 1], 'labels': {0: 'No', 1: 'Yes'}}
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
            # 修正Hospital Grade选择框，添加format_func显示英文标签
            input_data['HospitalGrade'] = st.selectbox(
                FEATURE_DETAILS['HospitalGrade']['display'],
                FEATURE_DETAILS['HospitalGrade']['options'],
                format_func=lambda x: FEATURE_DETAILS['HospitalGrade']['labels'][x],
                index=FEATURE_DETAILS['HospitalGrade']['options'].index(FEATURE_DETAILS['HospitalGrade']['default'])
            )
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
            input_data['BowelMovements'] = st.selectbox(FEATURE_DETAILS['BowelMovements']['display'], FEATURE_DETAILS['BowelMovements']['options'], format_func=lambda x: FEATURE_DETAILS['BowelMovements']['labels'][x])

        # Column 2: Preparation & Surgery History
        with cols_main[1]:
            st.markdown("**Patient Education**")
            # 修正BP Education Modality选择框，显示英文标签
            input_data['BPEducationModality'] = st.selectbox(
                FEATURE_DETAILS['BPEducationModality']['display'],
                FEATURE_DETAILS['BPEducationModality']['options'],
                format_func=lambda x: FEATURE_DETAILS['BPEducationModality']['labels'][x],
                index=FEATURE_DETAILS['BPEducationModality']['options'].index(FEATURE_DETAILS['BPEducationModality']['default'])
            )

            input_data['PreColonoscopyPhysicalActivity'] = st.selectbox(FEATURE_DETAILS['PreColonoscopyPhysicalActivity']['display'], FEATURE_DETAILS['PreColonoscopyPhysicalActivity']['options'], format_func=lambda x: FEATURE_DETAILS['PreColonoscopyPhysicalActivity']['labels'][x])
            
            st.markdown("**Previous Surgery History**")
            input_data['PreviousAbdominopelvicSurgery_1.0'] = st.selectbox(FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['display'], FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['options'], format_func=lambda x: FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['labels'][x], index=FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['options'].index(FEATURE_DETAILS['PreviousAbdominopelvicSurgery_1.0']['default']))
            input_data['PreviousAbdominopelvicSurgery_2.0'] = st.selectbox(FEATURE_DETAILS['PreviousAbdominopelvicSurgery_2.0']['display'], FEATURE_DETAILS['PreviousAbdominopelvicSurgery_2.0']['options'], format_func=lambda x: FEATURE_DETAILS['PreviousAbdominopelvicSurgery_2.0']['labels'][x])
            input_data['PreviousAbdominopelvicSurgery_3.0'] = st.selectbox(FEATURE_DETAILS['PreviousAbdominopelvicSurgery_3.0']['display'], FEATURE_DETAILS['PreviousAbdominopelvicSurgery_3.0']['options'], format_func=lambda x: FEATURE_DETAILS['PreviousAbdominopelvicSurgery_3.0']['labels'][x])
            
            st.markdown("**Diet Restriction**")
            cols_diet = st.columns(2)
            with cols_diet[0]:
                input_data['DietaryRestriction_1'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_1']['display'], FEATURE_DETAILS['DietaryRestriction_1']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_1']['labels'][x])
                input_data['DietaryRestriction_2'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_2']['display'], FEATURE_DETAILS['DietaryRestriction_2']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_2']['labels'][x])
            with cols_diet[1]:
                input_data['DietaryRestriction_3'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_3']['display'], FEATURE_DETAILS['DietaryRestriction_3']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_3']['labels'][x])
                input_data['DietaryRestriction_4'] = st.selectbox(FEATURE_DETAILS['DietaryRestriction_4']['display'], FEATURE_DETAILS['DietaryRestriction_4']['options'], format_func=lambda x: FEATURE_DETAILS['DietaryRestriction_4']['labels'][x])
               
            input_data['DietaryRestrictionDays'] = st.slider(FEATURE_DETAILS['DietaryRestrictionDays']['display'], FEATURE_DETAILS['DietaryRestrictionDays']['min'], FEATURE_DETAILS['DietaryRestrictionDays']['max'], FEATURE_DETAILS['DietaryRestrictionDays']['default'])
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
def generate_counterfactuals(model, patient_data):
    # ========== 加载真实训练数据（确保train_data.csv包含所有特征列 + outcome列） ==========
    # 直接加载真实训练数据，列顺序与你原代码严格对齐
    train_data = pd.read_csv("train_data.csv")  
    train_data.columns = ['HospitalGrade', 'Age', 'Sex', 'BMI', 'InpatientStatus', 'PreviousColonoscopy', 'ChronicConstipation', 
                          'DiabetesMellitus', 'StoolForm', 'BowelMovements', 'BPEducationModality', 'DietaryRestrictionDays', 
                          'PreColonoscopyPhysicalActivity', 'PreviousAbdominopelvicSurgery_1.0', 'PreviousAbdominopelvicSurgery_2.0', 
                          'PreviousAbdominopelvicSurgery_3.0','DietaryRestriction_1', 'DietaryRestriction_2', 'DietaryRestriction_3',
                          'DietaryRestriction_4', 'LaxativeRegimen_1', 'LaxativeRegimen_2', 'LaxativeRegimen_3', 'LaxativeRegimen_4', 
                          'LaxativeRegimen_5', 'LaxativeRegimen_6', 'outcome']
    st.success("")
    
    # ========== 与你原代码一致的特征类型定义 ==========
    continuous_vars = ['Age', 'BMI', 'DietaryRestrictionDays']  # 与你原代码完全一致
    binary_vars = ['HospitalGrade','Sex', 'InpatientStatus','PreviousColonoscopy', 
                   'ChronicConstipation', 'DiabetesMellitus', 'StoolForm', 'BPEducationModality', 
                   'PreColonoscopyPhysicalActivity', 'outcome',
                   'PreviousAbdominopelvicSurgery_1.0','PreviousAbdominopelvicSurgery_2.0',
                   'PreviousAbdominopelvicSurgery_3.0','DietaryRestriction_1', 'DietaryRestriction_2', 
                   'DietaryRestriction_3','DietaryRestriction_4', 'LaxativeRegimen_1', 'LaxativeRegimen_2',
                   'LaxativeRegimen_3', 'LaxativeRegimen_4', 'LaxativeRegimen_5','LaxativeRegimen_6']
    ordinal_vars = ['BowelMovements']  # 与你原代码完全一致
    
    # ========== 严格对齐你原代码的可干预变量（features_to_vary） ==========
    INTERVENABLE_FEATURES = [
        'DietaryRestrictionDays', 
        'PreColonoscopyPhysicalActivity',
        'DietaryRestriction_1', 'DietaryRestriction_2', 'DietaryRestriction_3', 'DietaryRestriction_4',
        'LaxativeRegimen_1', 'LaxativeRegimen_2', 'LaxativeRegimen_3', 'LaxativeRegimen_4', 'LaxativeRegimen_5', 'LaxativeRegimen_6'
    ]
    INTERVENABLE_FEATURES = [col for col in INTERVENABLE_FEATURES if col in patient_data.columns]
    
    # ========== 创建DiCE对象（与你原代码参数一致） ==========
    data = dice_ml.Data(
        dataframe=train_data, 
        continuous_features=continuous_vars, 
        categorical_features=binary_vars + ordinal_vars,
        outcome_name='outcome'
    )
    wrapped_model = ModelWrapper(model, threshold=0.27)  # 与你原阈值一致
    dice_model = dice_ml.Model(model=wrapped_model, backend='sklearn')
    exp = dice_ml.Dice(data, dice_model)
    
    # ========== 与你原代码完全一致的互斥约束（必须恰好1个） ==========
    def mutually_exclusive_constraint(instance):
        # 检查dietary变量：必须恰好1个为1
        dietary_values = [instance['DietaryRestriction_1'], instance['DietaryRestriction_2'], 
                          instance['DietaryRestriction_3'], instance['DietaryRestriction_4']]
        dietary_sum = sum(dietary_values)
        if dietary_sum != 1:
            return False
        
        # 检查protocol变量：必须恰好1个为1
        protocol_values = [instance['LaxativeRegimen_1'], instance['LaxativeRegimen_2'], 
                           instance['LaxativeRegimen_3'], instance['LaxativeRegimen_4'],
                           instance['LaxativeRegimen_5'], instance['LaxativeRegimen_6']]
        protocol_sum = sum(protocol_values)
        if protocol_sum != 1:
            return False
        
        return True
    
    # ========== 与你原代码一致的临床合理性检查 ==========
    def clinical_plausibility_check(cf_instance, original_instance):
        issues = []
        # 年龄不能减少（与你原代码一致）
        if cf_instance['Age'] < original_instance['Age'].values[0]:
            issues.append("年龄不能减少")
        
        # BMI变化不超过5（与你原代码一致）
        bmi_change = abs(cf_instance['BMI'] - original_instance['BMI'].values[0])
        if bmi_change > 5:
            issues.append(f"BMI变化过大: {bmi_change:.2f}")
        
        return issues
    
    # ========== 生成反事实（变量范围与你原代码一致） ==========
    try:
        dice_exp = exp.generate_counterfactuals(
            patient_data,
            total_CFs=10,  # 与你原代码数量一致
            features_to_vary=INTERVENABLE_FEATURES,  # 严格对齐你指定的变量
            desired_class="opposite"  # 指定目标类为0（与你原逻辑一致）
        )
        
        # ========== 与你原代码一致的筛选逻辑 ==========
        def filter_counterfactuals(cf_df):
            filtered_cfs = []
            for _, cf in cf_df.iterrows():
                # 检查互斥约束
                if mutually_exclusive_constraint(cf):
                    # 检查临床合理性
                    issues = clinical_plausibility_check(cf, patient_data)
                    if not issues:
                        filtered_cfs.append(cf)
                    else:
                        st.write(f"排除一个反事实，原因: {', '.join(issues)}")
            return pd.DataFrame(filtered_cfs)
        
        # 获取反事实结果并筛选
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        filtered_cf_df = filter_counterfactuals(cf_df)
        
        # 生成解释（与你原代码逻辑一致）
        explanations = []
        if not filtered_cf_df.empty:
            for idx, cf in filtered_cf_df.iterrows():
                changes = {}
                for col in cf.index:
                    if col != 'outcome' and cf[col] != patient_data[col].values[0]:
                        display_name = FEATURE_DETAILS.get(col, {}).get('display', col)
                        label_map = FEATURE_DETAILS.get(col, {}).get('labels', {})

              # 原始值和新值
                        orig_val = patient_data[col].values[0]
                        new_val = cf[col]
                        # 映射为标签（如果有labels字典，否则保持原值）
                        orig_label = label_map.get(orig_val, orig_val)
                        new_label = label_map.get(new_val, new_val)

                # 使用显示名称作为key，标签作为值
                       changes[display_name] = f"{orig_label} → {new_label}"
                    
                explanations.append(changes)
        
        return filtered_cf_df, explanations
    
    except Exception as e:
        st.error(f"生成反事实时出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
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
    st.title("Colonoscopy Bowel Preparation Quality Predictor")
    st.markdown("""
This is a tool designed to predict the quality of bowel preparation before a colonoscopy. It was developed using data from more than 12,000 
colonoscopy patients across over 170 hospitals nationwide. Based on a random forest model, the tool predicts a patient’s risk level (high/low) 
of inadequate bowel preparation using 16 features: age, sex, BMI, chronic constipation, diabetes mellitus, previous surgery history, previous
 colonoscopy, inpatient status, bowel preparation education modality, diet restriction type, dietary restriction days, laxative regimen, bowel 
movements, stool form, pre-colonoscopy physical activity and hospital grade. Additionally, the tool incorporates a counterfactual module 
that provides targeted improvement suggestions for clinicians and patients to help reduce the risk of preparation inadequate. 
   """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create input form
    patient_data = create_input_form()
    
    if patient_data is not None:
               
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
                    st.markdown(f"**Counterfactual {i+1}**")  # 反事实改为英文
                    # 遍历当前反事实的特征变化
                    for feature, change in explanations[i].items():
                        st.write(f"- {feature}: {change}")
                    # 定义cf_display并展示
                    #cf_display = cf[FEATURE_ORDER]  # 直接取反事实的所有特征列
                    #cf_display.index = [FEATURE_DETAILS[col]['display'] for col in cf_display.index]
                    #st.dataframe(cf_display.T, use_container_width=True)
            else:
                st.info("No valid counterfactual recommendations found.")

if __name__ == "__main__":
    main()






