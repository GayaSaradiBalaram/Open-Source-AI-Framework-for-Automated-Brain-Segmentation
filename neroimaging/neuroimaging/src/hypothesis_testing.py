import pandas as pd
import scipy.stats as stats

# ✅ Load Dice Scores from Segmentation Evaluation
segmentation_scores = {
    "Original": [0.85, 0.86, 0.84, 0.83, 0.87],  # Example Dice scores
    "Augmented": [0.88, 0.89, 0.87, 0.85, 0.90]  # Improved scores with augmentation
}

# ✅ Convert to DataFrame
df = pd.DataFrame(segmentation_scores)

# ✅ Perform Paired t-Test (Compare Original vs Augmented)
t_stat, p_value = stats.ttest_rel(df["Original"], df["Augmented"])
print(f"**Paired t-Test Results**: t-stat = {t_stat:.3f}, p-value = {p_value:.5f}")

# ✅ Perform ANOVA (if multiple model variations exist)
f_stat, anova_p = stats.f_oneway(df["Original"], df["Augmented"])
print(f"**ANOVA Results**: F-stat = {f_stat:.3f}, p-value = {anova_p:.5f}")

# ✅ Interpretation
if p_value < 0.05:
    print("The improvement in segmentation accuracy is statistically significant!")
else:
    print("No significant difference found. Consider fine-tuning further.")
