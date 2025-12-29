# Model Card: Heart Disease Prediction Model

## Model Details

| Property | Value |
|----------|-------|
| Model Name | Heart Disease Prediction Model |
| Model Version | 1.0.0 |
| Model Type | Random Forest Classifier |
| Framework | scikit-learn 1.7.2 |
| Training Date | December 2025 |
| Developed By | Emmanuel Chinonso Nwanguma |

## Intended Use

### Primary Use Cases

- **Clinical Decision Support**: Assist healthcare professionals in identifying patients at risk of heart disease for further evaluation
- **Health Screening**: Preliminary risk assessment tool for health checkups and wellness programs
- **Educational Purposes**: Demonstrate ML deployment practices and explainable AI techniques

### Out-of-Scope Uses

- **Diagnostic Tool**: This model is NOT intended to diagnose heart disease. It should only be used as a screening aid
- **Emergency Decisions**: Should NOT be used for emergency medical decisions
- **Standalone Medical Advice**: Must always be used in conjunction with professional medical evaluation
- **Pediatric Patients**: Model was trained on adult data (ages 29-77) and should not be used for patients under 20

## Model Performance

### Test Set Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.885 |
| ROC-AUC | 0.954 |
| Precision | 0.862 |
| Recall | 0.893 |
| F1 Score | 0.877 |

### Performance Interpretation

- **Accuracy (88.5%)**: The model correctly classifies 88.5% of patients
- **ROC-AUC (0.954)**: Excellent discrimination ability between positive and negative cases
- **Recall (89.3%)**: The model identifies 89.3% of actual heart disease cases (important for screening)
- **Precision (86.2%)**: 86.2% of positive predictions are correct

### Risk Level Thresholds

| Risk Level | Probability Range | Interpretation |
|------------|-------------------|----------------|
| Low | < 30% | Lower likelihood of heart disease |
| Medium | 30% - 70% | Moderate risk, further evaluation recommended |
| High | ≥ 70% | Higher likelihood, medical consultation advised |

## Training Data

### Dataset Overview

| Property | Value |
|----------|-------|
| Source | UCI Heart Disease Dataset (Cleveland) |
| Total Samples | 303 |
| Training Samples | ~242 (80%) |
| Test Samples | ~61 (20%) |
| Features | 13 clinical attributes |
| Target | Binary (0 = No heart disease, 1 = Heart disease) |

### Feature Description

| Feature | Description | Range/Values |
|---------|-------------|--------------|
| age | Age in years | 20-100 |
| sex | Biological sex | 0=Female, 1=Male |
| cp | Chest pain type | 1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic |
| trestbps | Resting blood pressure (mm Hg) | 80-200 |
| chol | Serum cholesterol (mg/dl) | 100-600 |
| fbs | Fasting blood sugar > 120 mg/dl | 0=No, 1=Yes |
| restecg | Resting ECG results | 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy |
| thalach | Maximum heart rate achieved | 60-220 |
| exang | Exercise induced angina | 0=No, 1=Yes |
| oldpeak | ST depression induced by exercise | 0.0-7.0 |
| slope | Slope of peak exercise ST segment | 1=Upsloping, 2=Flat, 3=Downsloping |
| ca | Number of major vessels colored by fluoroscopy | 0-3 |
| thal | Thalassemia | 3=Normal, 6=Fixed defect, 7=Reversible defect |

### Derived Features

The model also uses engineered features:
- **age_group**: Categorical age grouping (0-3)
- **chol_risk**: High cholesterol indicator (chol > 240)
- **bp_risk**: High blood pressure indicator (trestbps > 140)
- **heart_rate_reserve**: Calculated as (220 - age - thalach)

## Limitations

### Data Limitations

1. **Sample Size**: Training data contains only 303 samples, which may limit generalization
2. **Geographic Bias**: Data collected from Cleveland Clinic may not represent global populations
3. **Temporal Bias**: Original data collected in 1988; medical practices and population health have evolved
4. **Age Range**: Training data primarily covers ages 29-77; predictions outside this range may be less reliable

### Technical Limitations

1. **Binary Classification**: Model only predicts presence/absence, not severity or type of heart disease
2. **Feature Availability**: Requires all 13 clinical features; missing data cannot be handled
3. **Static Prediction**: Does not account for temporal changes in patient health

### Known Biases

1. **Gender Imbalance**: Dataset contains more male patients (~68%) than female (~32%)
2. **Age Distribution**: Majority of patients are between 40-65 years old
3. **Ethnicity**: Dataset lacks diversity in ethnic representation (primarily Caucasian patients)

## Ethical Considerations

### Fairness

- Model performance should be monitored across demographic groups
- Healthcare providers should be aware of potential disparities in predictions
- Regular audits recommended to assess fairness metrics

### Privacy

- Model requires sensitive health information
- All patient data should be handled according to HIPAA guidelines
- No patient data is stored by the model after prediction

### Transparency

- SHAP values provided for every prediction to explain feature contributions
- Model architecture and training process fully documented
- Open-source implementation allows for independent verification

## Recommendations

### For Healthcare Providers

1. Use predictions as one input among many in clinical decision-making
2. Always combine with patient history, physical examination, and other diagnostic tests
3. Consider the confidence level (probability) when interpreting results
4. Review SHAP explanations to understand which factors drove the prediction

### For Developers

1. Monitor model performance in production with real-world data
2. Implement logging to track prediction distributions over time
3. Plan for periodic model retraining with updated data
4. Consider A/B testing before deploying model updates

## Model Updates

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | December 2025 | Initial release with Random Forest classifier |

## Citation

If using this model in research or publications, please cite:

```
Heart Disease Prediction Model v1.0.0
Emmanuel Chinonso Nwanguma, 2025
Based on UCI Heart Disease Dataset (Cleveland)
```

## Contact

For questions, issues, or feedback about this model:

- **LinkedIn**: [linkedin.com/in/nwangumaemmanuel](https://linkedin.com/in/nwangumaemmanuel)
- **Email**: nwangumaemmanuel29@gmail.com
- **GitHub Issues**: Open an issue in the project repository
